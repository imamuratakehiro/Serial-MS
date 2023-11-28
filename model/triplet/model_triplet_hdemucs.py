"""640次元をcsnでマスク、それぞれをDecode。Decoderは5つ"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import matplotlib.pyplot as plt
import torchaudio
import math
import numpy as np
import os
#import stempeg
import csv
import pandas as pd
#import soundfile

from utils.func import progress_bar, normalize, denormalize
from ..csn import ConditionalSimNet2d
from ..to1d.model_linear import To1D640Demucs

from ..csn import ConditionalSimNet2d, ConditionalSimNet1d

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass

def rescale_conv(conv, reference):
    """Rescale initial weight scale. It is unclear why it helps but it certainly does.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)

def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.

    This will pad the input so that `F = ceil(T / K)`.

    see https://github.com/pytorch/pytorch/issues/60466
    つまりinputが[B, Ch, T = 1000]で、K = 200でsplitするとしたら、[B, Ch, F = 5, K = 200]が出力される。
    splitされるのは最後の次元。
    足りない分は何かで埋めてる、F.Pad参照
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, 'data should be contiguous'
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)

class LayerScale(nn.Module):
    """Layer scale from [Touvron et al 2021] (https://arxiv.org/pdf/2103.17239.pdf).
    This rescales diagonaly residual outputs close to 0 initially, then learnt.
    channelごとにinitの値をかける処理を施す。
    initの値を最初1e-3のような小さな値にすることで残差ブロックがじわじわ効いてくることを期待する。
    """

    def __init__(self, channels: int, init: float = 0, channel_last=False):
        """
        channel_last = False corresponds to (B, C, T) tensors
        channel_last = True corresponds to (T, B, C) tensors
        """
        super().__init__()
        self.channel_last = channel_last
        self.scale = nn.Parameter(torch.zeros(channels, requires_grad=True))
        self.scale.data[:] = init

    def forward(self, x):
        if self.channel_last:
            return self.scale * x
        else:
            return self.scale[:, None] * x

class BiLSTM(nn.Module):
    def __init__(self, dim, max_steps=200) -> None:
        super().__init__()
        self.blstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*dim, dim)
        self.max_steps = max_steps
    
    def forward(self, input):
        # max_stepsでsplit
        x = input.clone()
        B, Ch, T = x.shape
        width = self.max_steps
        stride = width // 2
        frames = unfold(x, width, stride) #最後に次元(時間)をmax_steps幅でsplitする。[B, Ch, frame数, max_steps]
        nframes = frames.shape[2]
        x = frames.permute(0, 2, 1, 3).reshape(-1, Ch, width) # [B, frame数, Ch, max_steps] -> [B×frame数, Ch, max_steps]

        # BiLSTM
        x = x.permute(0, 2, 1) # 文の長さ = max_steps、ベクトル長 = ch数
        x = self.blstm(x)[0]
        x = self.linear(x)
        x = x.permute(0, 2, 1) # [B×frame数, Ch, max_steps]

        # splitを復元
        out = []
        frames = x.reshape(B, -1, Ch, width) # [B, frame数, Ch, max_steps]
        limit = stride // 2
        # strideの半分のところでframeを繋ぎ合わせる
        for k in range(nframes):
            if k == 0:
                out.append(frames[:, k, :, :-limit])
            elif k == nframes - 1:
                out.append(frames[:, k, :, limit:])
            else:
                out.append(frames[:, k, :, limit:-limit])
        out = torch.cat(out, -1) # [B, Ch, T+paddingした長さ]
        out = out[..., :T] # [B, Ch, T]
        #output = input + out
        return out


class LocalState(nn.Module):
    """Local state allows to have attention based only on data (no positional embedding),
    but while setting a constraint on the time window (e.g. decaying penalty term).

    Also a failed experiments with trying to provide some frequency based attention.
    """
    def __init__(self, channels: int, heads: int = 4, nfreqs: int = 0, ndecay: int = 4):
        super().__init__()
        assert channels % heads == 0, (channels, heads)
        self.heads = heads
        self.nfreqs = nfreqs
        self.ndecay = ndecay
        self.content = nn.Conv1d(channels, channels, 1)
        self.query = nn.Conv1d(channels, channels, 1)
        self.key = nn.Conv1d(channels, channels, 1)
        if nfreqs:
            self.query_freqs = nn.Conv1d(channels, heads * nfreqs, 1)
        if ndecay:
            self.query_decay = nn.Conv1d(channels, heads * ndecay, 1)
            # Initialize decay close to zero (there is a sigmoid), for maximum initial window.
            self.query_decay.weight.data *= 0.01
            assert self.query_decay.bias is not None  # stupid type checker
            self.query_decay.bias.data[:] = -2
        self.proj = nn.Conv1d(channels + heads * nfreqs, channels, 1)

    def forward(self, x):
        B, C, T = x.shape
        heads = self.heads
        indexes = torch.arange(T, device=x.device, dtype=x.dtype)
        # left index are keys, right index are queries
        delta = indexes[:, None] - indexes[None, :]

        queries = self.query(x).view(B, heads, -1, T)
        keys = self.key(x).view(B, heads, -1, T)
        # t are keys, s are queries
        dots = torch.einsum("bhct,bhcs->bhts", keys, queries)
        dots /= keys.shape[2]**0.5
        if self.nfreqs:
            periods = torch.arange(1, self.nfreqs + 1, device=x.device, dtype=x.dtype)
            freq_kernel = torch.cos(2 * math.pi * delta / periods.view(-1, 1, 1))
            freq_q = self.query_freqs(x).view(B, heads, -1, T) / self.nfreqs ** 0.5
            dots += torch.einsum("fts,bhfs->bhts", freq_kernel, freq_q)
        if self.ndecay:
            decays = torch.arange(1, self.ndecay + 1, device=x.device, dtype=x.dtype)
            decay_q = self.query_decay(x).view(B, heads, -1, T)
            decay_q = torch.sigmoid(decay_q) / 2
            decay_kernel = - decays.view(-1, 1, 1) * delta.abs() / self.ndecay**0.5
            dots += torch.einsum("fts,bhfs->bhts", decay_kernel, decay_q)

        # Kill self reference.
        dots.masked_fill_(torch.eye(T, device=dots.device, dtype=torch.bool), -100)
        weights = torch.softmax(dots, dim=2)

        content = self.content(x).view(B, heads, -1, T)
        result = torch.einsum("bhts,bhct->bhcs", weights, content)
        if self.nfreqs:
            time_sig = torch.einsum("bhts,fts->bhfs", weights, freq_kernel)
            result = torch.cat([result, time_sig], 2)
        result = result.reshape(B, -1, T)
        return x + self.proj(result)


class ResidualBranch(nn.Module):
    def __init__(self, ch, dilation, lsat=False) -> None:
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv1d(ch, int(ch/4), kernel_size=3, stride=1, padding=dilation*(3//2), dilation=dilation),
            nn.GroupNorm(1, int(ch/4)), #Layer Normalization
            nn.GELU(),
        )
        if lsat:
            self.blstm_attention = nn.Sequential(
                BiLSTM(int(ch/4), max_steps=200),
                LocalState(int(ch/4), heads=4, ndecay=4),
            )
        self.conv = nn.Sequential(
            nn.Conv1d(int(ch/4), 2*ch, kernel_size=1, stride=1),
            nn.GroupNorm(1, 2*ch), #Layer Normalization
            nn.GLU(dim=1),
            LayerScale(ch, 1e-3),
        )
        self.lsat = lsat

    def forward(self, input):
        x = input.clone()
        dconv = self.dconv(x)
        if self.lsat:
            dconv = self.blstm_attention(dconv)
        conv = self.conv(dconv)
        skip = input + conv # skip接続
        return skip


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, lsat=False) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=8, stride=4, padding=padding),
            nn.GELU(),
        )
        self.dconv1 = ResidualBranch(out_channels, dilation=1, lsat=lsat)
        self.dconv2 = ResidualBranch(out_channels, dilation=2, lsat=lsat)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, 2*out_channels, kernel_size =1, stride=1),
            nn.GLU(dim=1),
        )
    def forward(self, input):
        """
        pad = input.shape[-1]%4
        print(input.shape)
        input = F.pad(input, (0,4-pad))
        print(input.shape)
        """
        conv1  = self.conv1(input)
        #print(conv1.shape)
        dconv1 = self.dconv1(conv1)
        dconv2 = self.dconv2(dconv1)
        conv2  = self.conv2(dconv2)
        return conv2

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, lsat=False) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(8,1), stride=(4,1), padding=(padding,0)),
            nn.GELU(),
        )
        self.dconv1 = ResidualBranch(out_channels, dilation=1, lsat=lsat)
        self.dconv2 = ResidualBranch(out_channels, dilation=2, lsat=lsat)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, 2*out_channels, kernel_size=(1,1), stride=(1,1)),
            nn.GLU(dim=1),
        )
    def forward(self, input):
        conv1 = self.conv1(input)
        #print(conv1.shape)
        B, Ch, Fr, T = conv1.shape
        conv1_time = conv1.permute(0,2,1,3).reshape(-1,Ch,T) #周波数方向はバッチ部分に拡張、つまりバッチ部分はB×Fr、[B, Fr, Ch, T] -> [B×Fr, Ch, T]
        dconv1_time = self.dconv1(conv1_time)
        dconv2_time = self.dconv2(dconv1_time)
        dconv2 = dconv2_time.reshape(B,Fr,Ch,T).permute(0,2,1,3) #周波数方向を元の位置に
        conv2 = self.conv2(dconv2)
        return conv2


class HDemucsTimeEncoder(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        # Encoder
        self.conv1 = Conv1d(input_ch, 48,  padding=2, lsat=False)
        self.conv2 = Conv1d(48,       96,  padding=2, lsat=False)
        self.conv3 = Conv1d(96,       192, padding=2, lsat=False)
        self.conv4 = Conv1d(192,      384, padding=2, lsat=False)
        self.conv5 = Conv1d(384,      768, padding=2, lsat=True)
        #self.conv6 = Conv1d(1024,     2048)
        #self.conv1 = Conv1d(input_ch, 16)
        #self.conv2 = Conv1d(16,       32)
        #elf.conv3 = Conv1d(32,      64)
        #self.conv4 = Conv1d(64,      128)
        #self.conv5 = Conv1d(128,      256)
        #self.conv6 = Conv1d(256,     512)
        #deviceを指定
        self.to(device)
    def forward(self, input):
        # Encoder
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        #conv6_out = self.conv6(conv5_out)
        return conv1_out, conv2_out, conv3_out, conv4_out, conv5_out#, conv6_out

class HDemucsFreqEncoder(nn.Module):
    """多分複素数で入力する"""
    def __init__(self, input_ch):
        super().__init__()
        # Encoder
        self.conv1 = Conv2d(input_ch, 48,  padding=2, lsat=False)
        self.conv2 = Conv2d(48,       96,  padding=2, lsat=False)
        self.conv3 = Conv2d(96,       192, padding=2, lsat=False)
        self.conv4 = Conv2d(192,      384, padding=2, lsat=False)
        self.conv5 = Conv2d(384,      768, padding=0, lsat=True)
        #self.conv6 = Conv2d(1024,     2048)
        #self.conv1 = Conv1d(input_ch, 16)
        #self.conv2 = Conv1d(16,       32)
        #elf.conv3 = Conv1d(32,      64)
        #self.conv4 = Conv1d(64,      128)
        #self.conv5 = Conv1d(128,      256)
        #self.conv6 = Conv1d(256,     512)
        #deviceを指定
        self.to(device)
    def forward(self, input):
        # Encoder
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        #conv6_out = self.conv6(conv5_out)
        return conv1_out, conv2_out, conv3_out, conv4_out, conv5_out#, conv6_out

class ConvTranspose1d(nn.Module):
    def __init__(self, in_channel, out_channel, padding, gl=True) -> None:
        super().__init__()
        #self.conv_1 = nn.Conv1d(2*in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv1d(in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1)
        self.glu    = nn.GLU(dim=1)
        self.conv_t = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=8, stride=4, padding=padding)
        if gl:
            self.gelu = nn.GELU()
        self.gl = gl

    def forward(self, input, encoder_out, shape):
        #conv_1 = self.conv_1(torch.cat([input, encoder_out], 1))
        if encoder_out is None:
            in_data = input
        else:
            in_data = input + encoder_out
        conv_1 = self.conv_1(in_data)
        glu    = self.glu(conv_1)
        conv_t = self.conv_t(glu, output_size=shape)
        if self.gl:
            output = self.gelu(conv_t)
        else:
            output = conv_t
        return output

class ConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding, gl=True) -> None:
        super().__init__()
        #self.conv_1 = nn.Conv1d(2*in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(in_channel, 2*in_channel, kernel_size=(3,1), stride=(1,1), padding=(1,0))
        self.glu    = nn.GLU(dim=1)
        self.conv_t = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=(8,1), stride=(4,1), padding=(padding,0))
        if gl:
            self.gelu = nn.GELU()
        self.gl = gl

    def forward(self, input, encoder_out, shape):
        #conv_1 = self.conv_1(torch.cat([input, encoder_out], 1))
        if encoder_out is None:
            in_data = input
        else:
            in_data = input + encoder_out
        conv_1 = self.conv_1(in_data)
        glu    = self.glu(conv_1)
        conv_t = self.conv_t(glu, output_size=shape)
        if self.gl:
            output = self.gelu(conv_t)
        else:
            output = conv_t
        return output


class HDemucsTimeDecoder(nn.Module):
    def __init__(self, output_ch) -> None:
        super().__init__()
        #self.deconv6 = ConvTranspose1d(768, 384,       gl=True)
        self.deconv5 = ConvTranspose1d(768, 384,       padding=2, gl=True)
        self.deconv4 = ConvTranspose1d(384, 192,       padding=2, gl=True)
        self.deconv3 = ConvTranspose1d(192, 96,        padding=2, gl=True)
        self.deconv2 = ConvTranspose1d(96,  48,        padding=2, gl=True)
        self.deconv1 = ConvTranspose1d(48,  output_ch, padding=2, gl=False)
        #self.deconv6 = ConvTranspose1d(512, 256,     rl=True)
        #self.deconv5 = ConvTranspose1d(256, 128,      rl=True)
        #self.deconv4 = ConvTranspose1d(128,  64,      rl=True)
        #self.deconv3 = ConvTranspose1d(64,  32,      rl=True)
        #self.deconv2 = ConvTranspose1d(32,  16,       rl=True)
        #self.deconv1 = ConvTranspose1d(16,  output_ch, rl=False)
    
    def forward(self, deconv6_out, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input):
        #deconv6_out = self.deconv6(conv6_out,   blstm_out, conv5_out.shape)
        deconv5_out = self.deconv5(deconv6_out, conv5_out, conv4_out.shape)
        deconv4_out = self.deconv4(deconv5_out, conv4_out, conv3_out.shape)
        deconv3_out = self.deconv3(deconv4_out, conv3_out, conv2_out.shape)
        deconv2_out = self.deconv2(deconv3_out, conv2_out, conv1_out.shape)
        deconv1_out = self.deconv1(deconv2_out, conv1_out, input.shape)
        return deconv1_out


class HDemucsFreqDecoder(nn.Module):
    def __init__(self, output_ch) -> None:
        super().__init__()
        #self.deconv6 = ConvTranspose1d(768, 384,       gl=True)
        self.deconv5 = ConvTranspose2d(768, 384,       padding=0, gl=True)
        self.deconv4 = ConvTranspose2d(384, 192,       padding=2, gl=True)
        self.deconv3 = ConvTranspose2d(192, 96,        padding=2, gl=True)
        self.deconv2 = ConvTranspose2d(96,  48,        padding=2, gl=True)
        self.deconv1 = ConvTranspose2d(48,  output_ch, padding=2, gl=False)
        #self.deconv6 = ConvTranspose1d(512, 256,     rl=True)
        #self.deconv5 = ConvTranspose1d(256, 128,      rl=True)
        #self.deconv4 = ConvTranspose1d(128,  64,      rl=True)
        #self.deconv3 = ConvTranspose1d(64,  32,      rl=True)
        #self.deconv2 = ConvTranspose1d(32,  16,       rl=True)
        #self.deconv1 = ConvTranspose1d(16,  output_ch, rl=False)
    
    def forward(self, deconv6_out, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input):
        #deconv6_out = self.deconv6(conv6_out,   blstm_out, conv5_out.shape)
        deconv5_out = self.deconv5(deconv6_out, conv5_out, conv4_out.shape)
        deconv4_out = self.deconv4(deconv5_out, conv4_out, conv3_out.shape)
        deconv3_out = self.deconv3(deconv4_out, conv3_out, conv2_out.shape)
        deconv2_out = self.deconv2(deconv3_out, conv2_out, conv1_out.shape)
        deconv1_out = self.deconv1(deconv2_out, conv1_out, input.shape)
        return deconv1_out


class TripletWithHDemucs(nn.Module):
    def __init__(self, inst_list, mono=True) -> None:
        super().__init__()
        #　Encoder
        if mono:
            self.input_ch = 1
        else:
            self.input_ch = 2
        # Encoder
        self.encoder_time = HDemucsTimeEncoder(input_ch=self.input_ch)
        self.encoder_freq = HDemucsFreqEncoder(input_ch=self.input_ch*2)
        self.encoder_6    = Conv1d(768, 1536, padding=2, lsat=True)
        # to1d
        self.to1d = To1D640Demucs(1536)
        # BiLSTM
        #self.blstm = BiLSTM(2048)
        #self.blstm = BiLSTM(512)
        # Decoder
        #self.decoder_6 = ConvTranspose1d(1536, 768, padding=2, gl=True)
        #self.decoder_time = HDemucsTimeDecoder(output_ch=len(inst_list)*self.input_ch)
        #self.decoder_freq = HDemucsFreqDecoder(output_ch=len(inst_list)*self.input_ch*2)
        # deviceを設定
        self.to(device)
        self.inst_list = inst_list
        rescale_module(self, reference=0.1)
    
    def spec(self, wave):
        *other, length = wave.shape
        x = wave.reshape(-1, length)
        n_fft = 4096
        z = torch.stft(x,
                        n_fft,
                        n_fft // 4,
                        window=torch.hann_window(n_fft).to(wave),
                        win_length=n_fft,
                        normalized=False,
                        center=True,
                        return_complex=True,
                        pad_mode='reflect')
        _, freqs, frame = z.shape
        return z.view(*other, freqs, frame)
    
    def magnitude(self, z):
        # return the magnitude of the spectrogram, except when cac is True,
        # in which case we just move the complex dimension to the channel one.
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3) # 複素数状態の虚数部分を別次元の実数として分ける
        m = m.reshape(B, C * 2, Fr, T)
        return m
    
    def normalize(self, data, dim):
        mean = data.mean(dim=dim, keepdim=True)
        std = data.std(dim=dim, keepdim=True)
        out = (data - mean) / (1e-5 + std)
        return out, mean, std
    
    def denormalize(self, data, mean, std):
        return data * (std - 1e-5) + mean
    
    def demagnitude(self, z_masc):
        out = z_masc.permute(0, 1, 2, 4, 5, 3)
        out = torch.view_as_complex(out.contiguous())
        return out
    
    def ispec(self, z):
        *other, freqs, frames = z.shape
        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)
        x = torch.istft(z,
                    n_fft,
                    n_fft // 4,
                    window=torch.hann_window(n_fft).to(z.real),
                    win_length=n_fft,
                    normalized=True,
                    #length=length,
                    center=True)
        _, length = x.shape
        return x.view(*other, length)
    
    def forward(self, input):
        # normalize
        #max = torch.max(input, dim=-1, keepdim=True).values
        #min = torch.min(input, dim=-1, keepdim=True).values
        #input = (input - min) / (max - min + 1e-5)
        #input_1024 = input[...,:input.shape[-1]//4096*4096+1]
        input_edited = F.pad(input, (0, 1024 - input.shape[-1]%1024))
        x_time = input_edited.clone()
        x_freq = self.magnitude(self.spec(input))
        Tt = x_time.shape[-1]
        B, C, Fq, Tf = x_freq.shape
        # Normalize
        x_time, mean_time, std_time = self.normalize(x_time, dim=(1,2))
        x_freq, mean_freq, std_freq = self.normalize(x_freq, dim=(1,2,3))
        # Encoder
        conv1_out_time, conv2_out_time, conv3_out_time, conv4_out_time, conv5_out_time = self.encoder_time(x_time)
        conv1_out_freq, conv2_out_freq, conv3_out_freq, conv4_out_freq, conv5_out_freq = self.encoder_freq(x_freq)
        conv6_out = self.encoder_6(torch.squeeze(conv5_out_freq) + conv5_out_time)
        output_emb = self.to1d(conv6_out)
        # BiLSTM
        #blstm_out = self.blstm(conv6_out)
        # 原点からのユークリッド距離にlogをかけてsigmoidしたものを無音有音の確率とする
        csn1d = ConditionalSimNet1d(); csn1d.to(output_emb.device)
        output_probability = {inst : torch.log(torch.sqrt(torch.sum(csn1d(output_emb, torch.tensor([i], device=output_emb.device))**2, dim=1))) for i,inst in enumerate(self.inst_list)} # logit
        """
        # Decoder
        deconv6_out = self.decoder_6(conv6_out, None, conv5_out_time.shape)
        output_time = self.decoder_time(deconv6_out, conv5_out_time, conv4_out_time, conv3_out_time, conv2_out_time, conv1_out_time, x_time)
        output_freq = self.decoder_freq(torch.unsqueeze(deconv6_out, dim=2), conv5_out_freq, conv4_out_freq, conv3_out_freq, conv2_out_freq, conv1_out_freq, x_freq)
        #print(output_time.shape)
        # Denormalize
        output_time = self.denormalize(output_time, mean=mean_time, std=std_time)
        output_freq = self.denormalize(output_freq, mean=mean_freq, std=std_freq)
        # reshape
        S = len(self.inst_list)
        output_time = output_time.view(B, S, self.input_ch, Tt)
        output_freq = output_freq.view(B, S, self.input_ch, 2, Fq, Tf)
        # freq -> wave
        output_zwave = self.ispec(self.demagnitude(output_freq))
        output = output_time[..., :output_zwave.shape[-1]] + output_zwave
        #output = output * (max - min + 1e-5) + min
        """
        return output_emb, output_probability, input



def main():
    # モデルを定義
    inst_list=["drums", "bass", "piano", "guitar", "residuals"]
    model = TripletWithHDemucs(inst_list=inst_list, mono=True)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 44100*10),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)
    
if "__main__" == __name__:
    main()
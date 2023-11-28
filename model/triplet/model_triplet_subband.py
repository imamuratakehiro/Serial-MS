"""Tripletのためのモデル。UNet部分は出力640次元を128次元で条件付けしてDecoderに入力。Decoderは5つ"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import os
#import stempeg
import csv
import pandas as pd
#import soundfile
import math
import typing as tp

from ..csn import ConditionalSimNet2d, ConditionalSimNet1d
from ..to1d.model_embedding import EmbeddingNet128to128, To1dEmbedding
from ..to1d.model_linear import To1D640
from utils.func import normalize_torch, denormalize_torch

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, last=False) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = (5, 5), stride=(2, 2), padding=2))
        #if not last:
        self.conv.add_module("bn", nn.BatchNorm2d(out_channels))
        self.conv.add_module("rl", nn.LeakyReLU(0.2))
    def forward(self, input):
        return self.conv(input)

class UNetEncoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size):
        super().__init__()
        # Encoder
        self.conv1 = Conv2d(encoder_in_size, 16)
        self.conv2 = Conv2d(16, 32)
        self.conv3 = Conv2d(32, 64)
        self.conv4 = Conv2d(64, 128)
        self.conv5 = Conv2d(128, 256)
        self.conv6 = Conv2d(256, encoder_out_size, last=True)
        #deviceを指定
        self.to(device)
    def forward(self, input):
        # Encoder
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        return conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out

def get_fftfreq(
        sr: int = 44100,
        n_fft: int = 2048
) -> torch.Tensor:
    """
    Torch workaround of librosa.fft_frequencies
    srとn_fftから求められるstftの結果の周波数メモリを配列にして出力。
    0から始まり、最後が22050。
    """
    out = sr * torch.fft.fftfreq(n_fft)[:n_fft // 2 + 1]
    out[-1] = sr // 2
    return out


def get_subband_indices(
        freqs: torch.Tensor,
        splits: tp.List[tp.Tuple[int, int]],
) -> tp.List[tp.Tuple[int, int]]:
    """
    Computes subband frequency indices with given bandsplits
    1. 入力で[end_freq, step]の組みが与えられる。
    2. stepの周波数幅でスペクトログラムをband splitする。
    3. end_freqがstepの値の区切り目で、達すると次のend_freqとstepに切り替わる。

    freqs_splits = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
    上記のsplitsが与えられた場合、
    start=0(Hz)     -> end=1000(Hz)の間を100(Hz)でband split
    start=1000(Hz)  -> end=4000(Hz)の間を250(Hz)でband split

    (以下略)

    start=16000(Hz) -> end=20000(Hz)の間を2000(Hz)でband split
    最後はstart=20000(Hz)で、残りの余った周波数部分を1bandとする。
    """
    indices = []
    start_freq, start_index = 0, 0
    for end_freq, step in splits:
        bands = torch.arange(start_freq + step, end_freq + step, step)
        start_freq = end_freq
        for band in bands:
            end_index = freqs[freqs < band].shape[0]
            indices.append((start_index, end_index))
            start_index = end_index
    indices.append((start_index, freqs.shape[0]))
    return indices

def freq2bands(
        bandsplits: tp.List[tp.Tuple[int, int]],
        sr: int = 44100,
        n_fft: int = 2048
) -> tp.List[tp.Tuple[int, int]]:
    """
    Returns start and end FFT indices of given bandsplits
    1. 入力で[end_freq, step]の組みが与えられる。
    2. stepの周波数幅でスペクトログラムをband splitする。
    3. end_freqがstepの値の区切り目で、達すると次のend_freqとstepに切り替わる。

    freqs_splits = [
        (1000, 100),
        (4000, 250),
        (8000, 500),
        (16000, 1000),
        (20000, 2000),
    ]
    上記のsplitsが与えられた場合、
    start=0(Hz)     -> end=1000(Hz)の間を100(Hz)でband split
    start=1000(Hz)  -> end=4000(Hz)の間を250(Hz)でband split

    (以下略)

    start=16000(Hz) -> end=20000(Hz)の間を2000(Hz)でband split
    最後はstart=20000(Hz)で、残りの余った周波数部分を1bandとする。
    """
    freqs = get_fftfreq(sr=sr, n_fft=n_fft)
    band_indices = get_subband_indices(freqs, bandsplits)
    return band_indices

class BandSplitModule(nn.Module):
    """
    BandSplit (1st) Module of BandSplitRNN.
    Separates input in k subbands and runs through LayerNorm+FC layers.
    """

    def __init__(
            self,
            cfg,
            sr: int,
            n_fft: int,
            #bandsplits: tp.List[tp.Tuple[int, int]],
            bandwidth_indices,
            fc_dim: int = 128,
            complex_as_channel: bool = True,
    ):
        super(BandSplitModule, self).__init__()

        frequency_mul = 1
        if complex_as_channel:
            frequency_mul *= 2
        if not cfg.mono:
            frequency_mul *= 2
        #print(is_mono)

        self.cac = complex_as_channel
        self.is_mono = cfg.mono
        #self.bandwidth_indices = freq2bands(bandsplits, sr, n_fft)
        self.bandwidth_indices = bandwidth_indices
        #self.layernorms = nn.ModuleList([
        #    nn.LayerNorm([(e - s) * frequency_mul, t_timesteps])
        #    for s, e in self.bandwidth_indices
        #])
        self.fcs = nn.ModuleList([
            nn.Linear((e - s) * frequency_mul, fc_dim)
            for s, e in self.bandwidth_indices
        ])

    def generate_subband(
            self,
            x: torch.Tensor
    ) -> tp.Iterator[torch.Tensor]:
        for start_index, end_index in self.bandwidth_indices:
            yield x[:, :, start_index:end_index]

    def forward(self, x: torch.Tensor):
        """
        Input: [batch_size, n_channels, freq, time]
        Output: [batch_size, k_subbands, time, fc_output_shape]
        """
        xs = []
        for i, x in enumerate(self.generate_subband(x)):
            B, C, F, T = x.shape
            # view complex as channels
            if x.dtype == torch.cfloat:
                x = torch.view_as_real(x).permute(0, 1, 4, 2, 3)
            # from channels to frequency
            x = x.reshape(B, -1, T) # [B, frequency_mul*subband_step, time]
            # run through model
            #x = self.layernorms[i](x)
            x = nn.LayerNorm(x.shape[-2:], elementwise_affine=False, device=x.device)(x)
            x = x.transpose(-1, -2) # [B, time, frequency_mul*subband_step]
            x = self.fcs[i](x) # [B, time, fc_dim]
            xs.append(x)
        return torch.stack(xs, dim=1).permute(0, 1, 3, 2)# [B, n_subbands, fc_dim, time]

class UNetForTriplet_Subband(nn.Module):
    def __init__(self, cfg, inst_list, f_size, mono=True, to1d_mode="mean_linear", order="timefreq", mel=False, n_mels=259):
        super().__init__()
        if mono:
            encoder_in_size = 1
        else:
            encoder_in_size = 2
        encoder_out_size = len(inst_list) * 128
        # Cul Subband_width
        fc_dim = 512
        bandsplits = [
            [500, 100],
            [4000, 1700],
            [17000, 5000]
        ]
        bandwidth_indices = freq2bands(bandsplits, cfg.sr, cfg.f_size)

        # encoder layer
        self.bandsplit = BandSplitModule(
            cfg=cfg,
            sr=cfg.sr,
            n_fft=cfg.f_size,
            #bandsplits=bandsplits,
            bandwidth_indices=bandwidth_indices,
            fc_dim=fc_dim,
            complex_as_channel=False,
        )
        # Encoder
        self.encoder = UNetEncoder(len(bandwidth_indices), encoder_out_size)
        if mel:
            #in_channel = (n_mels//(2**6)+1)*encoder_out_size
            in_channel = math.ceil(n_mels/(2**6))*encoder_out_size
        else:
            in_channel = (fc_dim/(2**6))*encoder_out_size
        self.to1d = nn.Sequential(
            nn.Linear(int(in_channel), encoder_out_size * 2),
            nn.ReLU(),
            nn.Linear(encoder_out_size * 2, encoder_out_size)
        )
        #deviceを指定
        self.inst_list = inst_list

    def forward(self, input):
        # Encoder
        input, max, min = normalize_torch(input)
        in_encoder = self.bandsplit(input)
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out = self.encoder(in_encoder)

        # to1d
        out_mean = torch.mean(conv6_out, dim=3)
        out_2d = out_mean.view(out_mean.size()[0], -1)
        output_emb = self.to1d(out_2d)
        # 原点からのユークリッド距離にlogをかけてsigmoidしたものを無音有音の確率とする
        csn1d = ConditionalSimNet1d() # csnのモデルを保存されないようにするために配列に入れる
        csn1d.to(output_emb.device)
        output_probability = {inst : torch.log(torch.sqrt(torch.sum(csn1d(output_emb, torch.tensor([i], device=device))**2, dim=1))) for i,inst in enumerate(self.inst_list)} # logit
        return output_emb, output_probability
def main():
    # モデルを定義
    inst_list = ["drums", "bass", "piano", "guitar", "residuals"]
    model = UNetForTriplet_2d_de5_embnet(inst_list=inst_list, f_size=1024)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 513, 259),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)
    
if "__main__" == __name__:
    main()
"""640次元をcsnでマスク、それぞれをDecode。Decoderは5つ"""

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

from utils.func import progress_bar, normalize, denormalize
from ..csn import ConditionalSimNet2d
from ..to1d.model_linear import To1D640Demucs

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
        #print(sub.__class__.__name__)
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            rescale_conv(sub, reference)


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(out_channels, 2*out_channels, kernel_size=1, stride=1),
            nn.GLU(dim=1),
        )
    def forward(self, input):
        return self.conv(input)

class DemucsEncoder(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        # Encoder
        self.conv1 = Conv1d(input_ch, 64)
        self.conv2 = Conv1d(64,       128)
        self.conv3 = Conv1d(128,      256)
        self.conv4 = Conv1d(256,      512)
        self.conv5 = Conv1d(512,      1024)
        self.conv6 = Conv1d(1024,     2048)
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
        conv6_out = self.conv6(conv5_out)
        return conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out

class BiLSTM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.blstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*dim, dim)
    
    def forward(self, input):
        x = input.permute(0, 2, 1) # 文の長さ = 時間長、ベクトル長 = ch数
        x = self.blstm(x)[0]
        x = self.linear(x)
        output = x.permute(0, 2, 1)
        return output

class ConvTranspose1d(nn.Module):
    def __init__(self, in_channel, out_channel, rl=True) -> None:
        super().__init__()
        #self.conv_1 = nn.Conv1d(2*in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv1d(in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1)
        self.glu    = nn.GLU(dim=1)
        self.conv_t = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=8, stride=4)
        if rl:
            self.relu   = nn.ReLU()
        self.rl = rl
        
    def forward(self, input, encoder_out, shape):
        #conv_1 = self.conv_1(torch.cat([input, encoder_out], 1))
        conv_1 = self.conv_1(input + encoder_out)
        glu    = self.glu(conv_1)
        conv_t = self.conv_t(glu, output_size=shape)
        if self.rl:
            output = self.relu(conv_t)
        else:
            output = conv_t
        return output


class DemucsDecoder(nn.Module):
    def __init__(self, output_ch) -> None:
        super().__init__()
        self.deconv6 = ConvTranspose1d(2048, 1024,     rl=True)
        self.deconv5 = ConvTranspose1d(1024, 512,      rl=True)
        self.deconv4 = ConvTranspose1d(512,  256,      rl=True)
        self.deconv3 = ConvTranspose1d(256,  128,      rl=True)
        self.deconv2 = ConvTranspose1d(128,  64,       rl=True)
        self.deconv1 = ConvTranspose1d(64,  output_ch, rl=False)
        #self.deconv6 = ConvTranspose1d(512, 256,     rl=True)
        #self.deconv5 = ConvTranspose1d(256, 128,      rl=True)
        #self.deconv4 = ConvTranspose1d(128,  64,      rl=True)
        #self.deconv3 = ConvTranspose1d(64,  32,      rl=True)
        #self.deconv2 = ConvTranspose1d(32,  16,       rl=True)
        #self.deconv1 = ConvTranspose1d(16,  output_ch, rl=False)
    
    def forward(self, blstm_out, conv6_out, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input):
        deconv6_out = self.deconv6(conv6_out,   blstm_out, conv5_out.shape)
        deconv5_out = self.deconv5(deconv6_out, conv5_out, conv4_out.shape)
        deconv4_out = self.deconv4(deconv5_out, conv4_out, conv3_out.shape)
        deconv3_out = self.deconv3(deconv4_out, conv3_out, conv2_out.shape)
        deconv2_out = self.deconv2(deconv3_out, conv2_out, conv1_out.shape)
        deconv1_out = self.deconv1(deconv2_out, conv1_out, input.shape)
        return deconv1_out


class TripletWithDemucs(nn.Module):
    def __init__(self, inst_list, mono=True) -> None:
        super().__init__()
        #　Encoder
        if mono:
            input_ch = 1
        else:
            input_ch = 2
        self.encoder = DemucsEncoder(input_ch=input_ch)
        # BiLSTM
        self.blstm = BiLSTM(2048)
        # to1d
        self.to1d = To1D640Demucs(2048)
        #self.blstm = BiLSTM(512)
        # Decoder
        #self.decoder = DemucsDecoder(output_ch=len(inst_list))
        # deviceを設定
        self.to(device)
        self.inst_list = inst_list
        rescale_module(self, reference=0.1)
    
    def forward(self, input):
        # normalize
        #max = torch.max(input, dim=-1, keepdim=True).values
        #min = torch.min(input, dim=-1, keepdim=True).values
        #input = (input - min) / (max - min + 1e-5)
        # Encoder
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out = self.encoder(input)
        # BiLSTM
        blstm_out = self.blstm(conv6_out)
        # to1d
        output = self.to1d(blstm_out)
        # Decoder
        #output = self.decoder(blstm_out, conv6_out, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input)
        #output = output * (max - min + 1e-5) + min
        return output



def main():
    # モデルを定義
    inst_list=0
    model = TripletWithDemucs(inst_list=inst_list, length=1024*100)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 1024*100),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)
    
if "__main__" == __name__:
    main()
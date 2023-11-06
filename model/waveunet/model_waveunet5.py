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

from ..csn import ConditionalSimNet2d

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(out_channels, 2*out_channels, kernel_size =3, stride=1, padding=1),
            nn.GLU(dim=1),
        )
    def forward(self, input):
        return self.conv(input)

class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = Conv1d(1, 16)
        self.conv2 = Conv1d(16, 32)
        self.conv3 = Conv1d(32, 64)
        self.conv4 = Conv1d(64, 128)
        self.conv5 = Conv1d(128, 256)
        self.conv6 = Conv1d(256, 512)
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

class ConvTranspose1d(nn.Module):
    def __init__(self, in_channel, out_channel, rl=True) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(2*in_channel, 2*in_channel, kernel_size=1, stride=1)
        self.glu    = nn.GLU(dim=1)
        self.conv_t = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=8, stride=4)
        if rl:
            self.relu   = nn.ReLU()
        self.rl = rl
        
    def forward(self, input, encoder_out, shape):
        conv_1 = self.conv_1(torch.cat([input, encoder_out], 1))
        glu    = self.glu(conv_1)
        conv_t = self.conv_t(glu, output_size=shape)
        if self.rl:
            output = self.relu(conv_t)
        else:
            output = conv_t
        return output

class ConvTranspose1d_NoCat(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channel, 2*in_channel, kernel_size=3, stride=1, padding=1)
        self.glu    = nn.GLU(dim=1)
        self.conv_t = nn.ConvTranspose1d(in_channel, out_channel, kernel_size=8, stride=4)
        self.relu   = nn.ReLU()
        
    def forward(self, input, shape):
        conv_1 = self.conv_1(input)
        glu    = self.glu(conv_1)
        conv_t = self.conv_t(glu, output_size=shape)
        relu   = self.relu(conv_t)
        return relu


class UNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deconv6 = ConvTranspose1d_NoCat(512, 256)
        self.deconv5 = ConvTranspose1d(256, 128, rl=True)
        self.deconv4 = ConvTranspose1d(128, 64, rl=True)
        self.deconv3 = ConvTranspose1d(64, 32, rl=True)
        self.deconv2 = ConvTranspose1d(32, 16, rl=True)
        self.deconv1 = ConvTranspose1d(16, 5, rl=False)
    
    def forward(self, conv6_out, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input):
        deconv6_out = self.deconv6(conv6_out, conv5_out.shape)
        deconv5_out = self.deconv5(deconv6_out, conv5_out, conv4_out.shape)
        deconv4_out = self.deconv4(deconv5_out, conv4_out, conv3_out.shape)
        deconv3_out = self.deconv3(deconv4_out, conv3_out, conv2_out.shape)
        deconv2_out = self.deconv2(deconv3_out, conv2_out, conv1_out.shape)
        deconv1_out = self.deconv1(deconv2_out, conv1_out, input.shape)
        return deconv1_out

"""
class UNetDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deconv6_b = nn.Sequential(
                                nn.Conv1d(in_channels, 2*out_channels, kernel_size = (8, 8), stride=(4, 4), padding=0),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv6_a = nn.ConvTranspose2d(640, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_a = nn.ConvTranspose2d(256+256, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_b = nn.Sequential(
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv4_a = nn.ConvTranspose2d(128+128, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_b = nn.Sequential(
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv3_a = nn.ConvTranspose2d(64+64, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_b = nn.Sequential(
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.2))
        self.deconv2_a = nn.ConvTranspose2d(32+32, 16, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_b = nn.Sequential(
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.2))
        self.deconv1_a = nn.ConvTranspose2d(16+16, 1, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv1 = nn.Sequential(
                                nn.LeakyReLU(0.2),
                                nn.Sigmoid())
        #deviceを指定
        self.to(device)
    def forward(self, sep_feature, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input):
        deconv6_out = self.deconv6_a(sep_feature, output_size = conv5_out.size())
        deconv6_out = self.deconv6_b(deconv6_out)
        deconv5_out = self.deconv5_a(torch.cat([deconv6_out, conv5_out], 1), output_size = conv4_out.size())
        deconv5_out = self.deconv5_b(deconv5_out)
        deconv4_out = self.deconv4_a(torch.cat([deconv5_out, conv4_out], 1), output_size = conv3_out.size())
        deconv4_out = self.deconv4_b(deconv4_out)
        deconv3_out = self.deconv3_a(torch.cat([deconv4_out, conv3_out], 1), output_size = conv2_out.size())
        deconv3_out = self.deconv3_b(deconv3_out)
        deconv2_out = self.deconv2_a(torch.cat([deconv3_out, conv2_out], 1), output_size = conv1_out.size())
        deconv2_out = self.deconv2_b(deconv2_out)
        deconv1_out = self.deconv1_a(torch.cat([deconv2_out, conv1_out], 1), output_size = input.size())
        output = torch.sigmoid(deconv1_out)
        return output
"""
class WaveUNet(nn.Module):
    def __init__(self, inst_list) -> None:
        super().__init__()
        #　Encoder
        self.encoder = UNetEncoder()
        # Decoder
        self.decoder = UNetDecoder()
        # deviceを設定
        self.to(device)
        self.inst_list = inst_list
    
    def forward(self, input):
        # Encoder
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out = self.encoder(input)
        # Decoder
        output = self.decoder(conv6_out, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input)
        return output

class UNetcsnde5(nn.Module):
    def __init__(self, inst_list):
        super().__init__()
        # Encoder
        self.encoder = UNetEncoder()
        # Decoder
        self.decoder_drums  = UNetDecoder()
        self.decoder_bass   = UNetDecoder()
        self.decoder_piano  = UNetDecoder()
        self.decoder_guitar = UNetDecoder()
        self.decoder_others = UNetDecoder()
        
        #deviceを指定
        self.to(device)
        self.inst_list = inst_list

    def forward(self, input):
        # Encoder
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out = self.encoder(input)

        # 特徴量を条件づけ
        size = conv6_out.shape
        # インスタンスを生成していなかったら、生成する。
        if not "csn" in vars(self):
            self.csn = ConditionalSimNet2d(size, device)

        # Decoder
        decoder = {
            "drums": self.decoder_drums,
            "bass": self.decoder_bass,
            "piano": self.decoder_piano,
            "guitar": self.decoder_guitar,
            "residuals": self.decoder_others,
        }
        output = {
            "drums": None,
            "bass": None,
            "piano": None,
            "guitar": None,
            "residuals": None,
        }
        for idx, inst in enumerate(self.inst_list):
            sep_feature = self.csn(conv6_out, torch.tensor([idx], device=device))  # 特徴量を条件づけ
            decoder_out = decoder[inst].forward(sep_feature, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input)
            output[inst] = decoder_out
        return output

def main():
    # モデルを定義
    inst_list=0
    model = WaveUNet(inst_list=inst_list)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 1024*100),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)
    
if "__main__" == __name__:
    main()
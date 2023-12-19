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

from ..csn import ConditionalSimNet2d, ConditionalSimNet1d
from ..to1d.model_embedding import EmbeddingNet128to128, To1dEmbedding
from ..to1d.model_linear import To1D640
from utils.func import normalize_torch, denormalize_torch, standardize_torch, destandardize_torch
from functools import partial

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass

def get_norm(norm_type):
    def norm(c, norm_type):   
        if norm_type=='BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type=='InstanceNorm':
            return nn.InstanceNorm2d(c, affine=True)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()
    return partial(norm, norm_type=norm_type) # norm関数のnorm_typeにすでに値が入った関数を返す


class DownScale2d(nn.Module):
    def __init__(self, in_channels, out_channels, last=False) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size = (5, 5), stride=(2, 2), padding=2)
            )
        #if not last:
        #self.conv.add_module("bn", nn.BatchNorm2d(out_channels))
        #self.conv.add_module("bn", nn.InstanceNorm2d(out_channels))
        #self.conv.add_module("rl", nn.LeakyReLU(0.2))
    def forward(self, input):
        return self.conv(input)

class TwoConvBlockSizeChange(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.twoconv = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding="same"),
            nn.InstanceNorm2d(middle_channels),
            nn.GELU(),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding="same"),
        )
        # サイズ変更ありのconvの場合、residuals接続方向もサイズ変更する必要あり。なので全結合と等価な1×1convを入れる。
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same")

    def forward(self, input):
        x = self.twoconv(input)
        return x + self.identity(input)

class TwoConvBlockAdditional(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.twoconv = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same"),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same"),
        )

    def forward(self, input):
        x = self.twoconv(input)
        return x + input

class ResConvBlock(nn.Module):
    def __init__(self, n_res, in_channels, middle_channels, out_channels) -> None:
        super().__init__()
        self.resblock = nn.ModuleList()
        for i in range(n_res):
            if i == 0:
                self.resblock.append(TwoConvBlockSizeChange(in_channels, middle_channels, out_channels))
            else:
                self.resblock.append(TwoConvBlockAdditional(out_channels))
    
    def forward(self, x):
        for block in self.resblock:
            x = block(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size, n_res):
        super().__init__()
        # Encoder
        self.first_conv = nn.Conv2d(encoder_in_size, encoder_in_size, kernel_size=1, padding="same")
        self.resconv1 = ResConvBlock(n_res, encoder_in_size, 32, 32)
        self.downscale1 = DownScale2d(32, 32)
        self.resconv2 = ResConvBlock(n_res, 32, 64, 64)
        self.downscale2 = DownScale2d(64, 64)
        self.resconv3 = ResConvBlock(n_res, 64, 128, 128)
        self.downscale3 = DownScale2d(128, 128)
        self.resconv4 = ResConvBlock(n_res, 128, 256, 256)
        self.downscale4 = DownScale2d(256, 256)
        self.resconv5 = ResConvBlock(n_res, 256, encoder_out_size, encoder_out_size)
        self.downscale5 = DownScale2d(encoder_out_size, encoder_out_size)
        #deviceを指定
        self.to(device)
    def forward(self, input):
        # Encoder
        first_conv = self.first_conv(input)
        resconv1_out = self.resconv1(first_conv)
        conv1_out = self.downscale1(resconv1_out)
        resconv2_out = self.resconv2(conv1_out)
        conv2_out = self.downscale2(resconv2_out)
        resconv3_out = self.resconv3(conv2_out)
        conv3_out = self.downscale3(resconv3_out)
        resconv4_out = self.resconv4(conv3_out)
        conv4_out = self.downscale4(resconv4_out)
        resconv5_out = self.resconv5(conv4_out)
        conv5_out = self.downscale5(resconv5_out)
        return conv1_out, conv2_out, conv3_out, conv4_out, conv5_out

class UNetDecoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size) -> None:
        super().__init__()
        #self.deconv6_a = nn.ConvTranspose2d(encoder_out_size, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        #self.deconv6_b = nn.Sequential(
        #                        #nn.BatchNorm2d(256),
        #                        nn.InstanceNorm2d(256),
        #                        nn.LeakyReLU(0.2),
        #                        nn.Dropout2d(0.5))
        self.deconv5_a = nn.ConvTranspose2d(encoder_out_size, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_b = nn.Sequential(
                                #nn.BatchNorm2d(128),
                                nn.InstanceNorm2d(256),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv4_a = nn.ConvTranspose2d(256+256, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_b = nn.Sequential(
                                #nn.BatchNorm2d(64),
                                nn.InstanceNorm2d(128),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv3_a = nn.ConvTranspose2d(128+128, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_b = nn.Sequential(
                                #nn.BatchNorm2d(32),
                                nn.InstanceNorm2d(64),
                                nn.LeakyReLU(0.2),)
        self.deconv2_a = nn.ConvTranspose2d(64+64, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_b = nn.Sequential(
                                #nn.BatchNorm2d(16),
                                nn.InstanceNorm2d(32),
                                nn.LeakyReLU(0.2),)
        self.deconv1_a = nn.ConvTranspose2d(32+32, encoder_in_size, kernel_size = (5, 5), stride=(2, 2), padding=2)
        #deviceを指定
        self.to(device)
    def forward(self, sep_feature, conv4_out, conv3_out, conv2_out, conv1_out, input):
        deconv5_out = self.deconv5_a(sep_feature, output_size = conv4_out.size())
        deconv5_out = self.deconv5_b(deconv5_out)
        #deconv5_out = self.deconv5_a(torch.cat([deconv6_out, conv5_out], 1), output_size = conv4_out.size())
        #deconv5_out = self.deconv5_b(deconv5_out)
        deconv4_out = self.deconv4_a(torch.cat([deconv5_out, conv4_out], 1), output_size = conv3_out.size())
        deconv4_out = self.deconv4_b(deconv4_out)
        deconv3_out = self.deconv3_a(torch.cat([deconv4_out, conv3_out], 1), output_size = conv2_out.size())
        deconv3_out = self.deconv3_b(deconv3_out)
        deconv2_out = self.deconv2_a(torch.cat([deconv3_out, conv2_out], 1), output_size = conv1_out.size())
        deconv2_out = self.deconv2_b(deconv2_out)
        deconv1_out = self.deconv1_a(torch.cat([deconv2_out, conv1_out], 1), output_size = input.size())
        output = torch.sigmoid(deconv1_out)
        return output

class UNetForTriplet_2d_de5_to1d64_resnet(nn.Module):
    def __init__(self, cfg, inst_list, f_size, mono=True, to1d_mode="mean_linear", order="timefreq", mel=False, n_mels=259):
        super().__init__()
        if mono:
            encoder_in_size = 1
        else:
            encoder_in_size = 2
        if cfg.complex:
            encoder_in_size *= 2
        encoder_out_size = len(inst_list) * 128
        # Encoder
        self.encoder = UNetEncoder(encoder_in_size, encoder_out_size, n_res=1)
        # Decoder
        """
        for inst in inst_list:
            if inst == "drums":
                self.decoder_drums = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "bass":
                self.decoder_bass = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "piano":
                self.decoder_piano = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "guitar":
                self.decoder_guitar = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "vocals":
                self.decoder_vocals = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "residuals":
                self.decoder_residuals = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "other":
                self.decoder_other = UNetDecoder(encoder_in_size, encoder_out_size)
        """
        # To1d
        #self.to1d = To1D640(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*encoder_out_size)
        if mel:
            #in_channel = (n_mels//(2**6)+1)*encoder_out_size
            in_channel = math.ceil(n_mels/(2**5))*encoder_out_size
        else:
            in_channel = (f_size/2/(2**5)+1)*encoder_out_size
        self.to1d = To1D640(to1d_mode=to1d_mode, order=order, in_channel=in_channel)
        #self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #deviceを指定
        self.to(device)
        self.inst_list = inst_list
        self.cfg = cfg

    def forward(self, input):
        # Encoder
        if self.cfg.standardize:
            input, mean, std = standardize_torch(input)
        elif self.cfg.normalize:
            input, max, min = normalize_torch(input)

        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out = self.encoder(input)

        # 特徴量を条件づけ
        size = conv5_out.shape
        #print(size)
        # インスタンスを生成していなかったら、生成する。
        csn = ConditionalSimNet2d(size, device) # csnのモデルを保存されないようにするために配列に入れる
        """
        decoder = {}
        output_decoder = {}
        for inst in self.inst_list:
            if inst == "drums":
                decoder[inst] = self.decoder_drums
            elif inst == "bass":
                decoder[inst] = self.decoder_bass
            elif inst == "piano":
                decoder[inst] = self.decoder_piano
            elif inst == "guitar":
                decoder[inst] = self.decoder_guitar
            elif inst == "vocals":
                decoder[inst] = self.decoder_vocals
            elif inst == "residuals":
                decoder[inst] = self.decoder_residuals
            elif inst == "other":
                decoder[inst] = self.decoder_other

        for idx, inst in enumerate(self.inst_list):
            # decoder
            sep_feature_decoder = csn(conv5_out, torch.tensor([idx], device=device))  # 特徴量を条件づけ
            decoder_out = decoder[inst].forward(sep_feature_decoder, conv4_out, conv3_out, conv2_out, conv1_out, input)
            if self.cfg.standardize:
                output_decoder[inst] = destandardize_torch(input*decoder_out, mean, std) # マスクをかけてからdestandardize
            elif self.cfg.normalize:
                output_decoder[inst] = denormalize_torch(input*decoder_out, max, min) # マスクをかけてからdenormalize
            else:
                output_decoder[inst] = input*decoder_out
        """
        # to1d
        output_emb = self.to1d(conv5_out)
        # 原点からのユークリッド距離にlogをかけてsigmoidしたものを無音有音の確率とする
        csn1d = ConditionalSimNet1d() # csnのモデルを保存されないようにするために配列に入れる
        csn1d.to(output_emb.device)
        output_probability = {inst : torch.log(torch.sqrt(torch.sum(csn1d(output_emb, torch.tensor([i], device=device))**2, dim=1))) for i,inst in enumerate(self.inst_list)} # logit
        return output_emb, output_probability#, output_decoder
    
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
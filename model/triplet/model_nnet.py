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

from utils.func import progress_bar
from ..csn import ConditionalSimNet2d
from ..to1d.model_embedding import EmbeddingNet128to128
from ..to1d.model_linear import To1D128timefreq, To1D128freqtime

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
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
        self.conv6 = Conv2d(256, encoder_out_size)
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

class NNetEncoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size) -> None:
        super().__init__()
        #Embedding Network
        self.emb_conv1 = Conv2d(encoder_in_size, 16)
        self.emb_conv2 = Conv2d(16+16, 32)
        self.emb_conv3 = Conv2d(32+32, 64)
        self.emb_conv4 = Conv2d(64+64, 128)
        self.emb_conv5 = Conv2d(128+128, 256)
        self.emb_conv6 = Conv2d(256+256, encoder_out_size)
        # UNet Encoder
        self.unet_conv1 = Conv2d(encoder_in_size, 16)
        self.unet_conv2 = Conv2d(16, 32)
        self.unet_conv3 = Conv2d(32, 64)
        self.unet_conv4 = Conv2d(64, 128)
        self.unet_conv5 = Conv2d(128, 256)
        self.unet_conv6 = Conv2d(256, encoder_out_size)
        
        #deviceを指定
        self.to(device)
    def forward(self, input):
        # Encoder
        unet_conv1_out = self.unet_conv1(input)
        unet_conv2_out = self.unet_conv2(unet_conv1_out)
        unet_conv3_out = self.unet_conv3(unet_conv2_out)
        unet_conv4_out = self.unet_conv4(unet_conv3_out)
        unet_conv5_out = self.unet_conv5(unet_conv4_out)
        unet_conv6_out = self.unet_conv6(unet_conv5_out)
        # Embedding Net
        emb_conv1_out  = self.emb_conv1(input)
        emb_conv2_out  = self.emb_conv2(torch.cat([unet_conv1_out, emb_conv1_out], 1))
        emb_conv3_out  = self.emb_conv3(torch.cat([unet_conv2_out, emb_conv2_out], 1))
        emb_conv4_out  = self.emb_conv4(torch.cat([unet_conv3_out, emb_conv3_out], 1))
        emb_conv5_out  = self.emb_conv5(torch.cat([unet_conv4_out, emb_conv4_out], 1))
        emb_conv6_out  = self.emb_conv6(torch.cat([unet_conv5_out, emb_conv5_out], 1))
        return unet_conv1_out, unet_conv2_out, unet_conv3_out, unet_conv4_out, unet_conv5_out, unet_conv6_out, emb_conv6_out

class UNetDecoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size) -> None:
        super().__init__()
        self.deconv6_a = nn.ConvTranspose2d(encoder_out_size, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv6_b = nn.Sequential(
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
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
        self.deconv1_a = nn.ConvTranspose2d(16+16, encoder_in_size, kernel_size = (5, 5), stride=(2, 2), padding=2)
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
        return deconv6_out, deconv5_out, deconv4_out, deconv3_out, deconv2_out, deconv1_out, output

class To1dEmbedding(nn.Module):
    def __init__(self, to1d_mode, order, in_channel, tanh = True) -> None:
        super().__init__()
        if order == "timefreq":
            self.embnet = nn.Sequential(To1D128timefreq(mode=to1d_mode, in_channel=int(in_channel)),
                                    EmbeddingNet128to128(tanh))
        elif order =="freqtime":
            self.embnet = nn.Sequential(To1D128freqtime(mode=to1d_mode, in_channel=int(in_channel)),
                                    EmbeddingNet128to128(tanh))
        self.recog = nn.Linear(128, 1)
        #deviceを指定
        self.to(device)
    def forward(self, input):
        emb_vec = self.embnet(input)
        recog_p = self.recog(emb_vec)
        return emb_vec, recog_p

class NNet(nn.Module):
    def __init__(self, inst_list, f_size, mono=True, to1d_mode="mean_linear", order="timefreq", tanh = True):
        super().__init__()
        if mono:
            encoder_in_size = 1
        else:
            encoder_in_size = 2
        encoder_out_size = len(inst_list) * 128
        # Encoder
        #self.unet_encoder = UNetEncoder(encoder_in_size, encoder_out_size)
        self.nnet_encoder = NNetEncoder(encoder_in_size, encoder_out_size)
        # Decoder・Embedding Network
        to1d_in_channel = 128
        for inst in inst_list:
            if inst == "drums":
                self.decoder_drums = UNetDecoder(encoder_in_size, encoder_out_size)
                self.embnet_drums  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*to1d_in_channel, tanh=tanh)
            elif inst == "bass":
                self.decoder_bass = UNetDecoder(encoder_in_size, encoder_out_size)
                self.embnet_bass  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*to1d_in_channel, tanh=tanh)
            elif inst == "piano":
                self.decoder_piano = UNetDecoder(encoder_in_size, encoder_out_size)
                self.embnet_piano  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*to1d_in_channel, tanh=tanh)
            elif inst == "guitar":
                self.decoder_guitar = UNetDecoder(encoder_in_size, encoder_out_size)
                self.embnet_guitar  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*to1d_in_channel, tanh=tanh)
            elif inst == "vocals":
                self.decoder_vocals = UNetDecoder(encoder_in_size, encoder_out_size)
                self.embnet_vocals  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*to1d_in_channel, tanh=tanh)
            elif inst == "residuals":
                self.decoder_residuals = UNetDecoder(encoder_in_size, encoder_out_size)
                self.embnet_residuals  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*to1d_in_channel, tanh=tanh)
            elif inst == "other":
                self.decoder_other = UNetDecoder(encoder_in_size, encoder_out_size)
                self.embnet_other  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*to1d_in_channel, tanh=tanh)
        #self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #deviceを指定
        self.to(device)
        self.inst_list = inst_list

    def forward(self, input):
        # Encoder
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out, emb_conv6_out = self.nnet_encoder(input)

        # 特徴量を条件づけ
        size = conv6_out.shape
        print(size)
        # インスタンスを生成していなかったら、生成する。
        if not "csn" in vars(self):
            self.csn = ConditionalSimNet2d(size, device)
        embnet = {}
        decoder = {}
        output_decoder = {}
        output_emb = {}
        output_probability = {}
        for inst in self.inst_list:
            if inst == "drums":
                embnet[inst] = self.embnet_drums
                decoder[inst] = self.decoder_drums
            elif inst == "bass":
                embnet[inst] = self.embnet_bass
                decoder[inst] = self.decoder_bass
            elif inst == "piano":
                embnet[inst] = self.embnet_piano
                decoder[inst] = self.decoder_piano
            elif inst == "guitar":
                embnet[inst] = self.embnet_guitar
                decoder[inst] = self.decoder_guitar
            elif inst == "vocals":
                embnet[inst] = self.embnet_vocals
                decoder[inst] = self.decoder_vocals
            elif inst == "residuals":
                embnet[inst] = self.embnet_residuals
                decoder[inst] = self.decoder_residuals
            elif inst == "other":
                embnet[inst] = self.embnet_other
                decoder[inst] = self.decoder_other

        for idx, inst in enumerate(self.inst_list):
            # decoder
            sep_feature_decoder = self.csn(conv6_out, torch.tensor([idx], device=device))  # 特徴量を条件づけ
            decoder_out = decoder[inst].forward(sep_feature_decoder, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input)
            output_decoder[inst] = decoder_out
            # embedding
            #emb, recog_probability = embnet[inst](torch.cat([conv6_out[:,128*idx : 128*(idx+1), :,:], emb_conv6_out[:,128*idx : 128*(idx+1), :,:]], dim=1))
            emb, recog_probability = embnet[inst](conv6_out[:,128*idx : 128*(idx+1), :,:])
            output_emb[inst] = emb
            # 原点からのユークリッド距離にtanhをしたものを無音有音の確率とする
            #output_probability[inst] = self.sigmoid(torch.log(torch.sqrt(torch.sum(emb**2, dim=1))))
            output_probability[inst] = self.sigmoid(recog_probability)[:,0]
            #print(output_probability[inst].shape)
        return output_emb, output_probability, output_decoder
    
def main():
    # モデルを定義
    inst_list = ["drums", "bass", "piano", "guitar", "residuals"]
    model = NNet(inst_list=inst_list, f_size=1024)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 513, 259),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)
    
if "__main__" == __name__:
    main()
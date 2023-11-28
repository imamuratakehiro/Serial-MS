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

class RNNModule(nn.Module):
    """
    RNN submodule of BandSequence module
    """

    def __init__(
            self,
            input_dim_size: int,
            hidden_dim_size: int,
            rnn_type: str = 'LSTM',
            bidirectional: bool = True,
            last: bool = False,
    ):
        super(RNNModule, self).__init__()
        self.groupnorm = nn.GroupNorm(input_dim_size, input_dim_size)
        self.rnn = getattr(nn, rnn_type)(
            input_dim_size, hidden_dim_size, batch_first=True, bidirectional=bidirectional
        )
        if not last:
            self.fc = nn.Linear(
                hidden_dim_size * 2 if bidirectional else hidden_dim_size,
                input_dim_size
            )
        self.last = last

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Input shape:
            across T - [batch_size, k_subbands, time, n_features]
            OR
            across K - [batch_size, time, k_subbands, n_features]
        """
        B, C, F, T = x.shape  # across T      across K (keep in mind T->K, K->T)

        out = x.view(B, C * F, T)  # [B, CF, T]

        out = self.groupnorm(out).transpose(-1, -2)  # [B, T, CF]
        h_all, (h, c) = self.rnn(out)  # [B, T, H*2] or [2, B, H]

        if self.last:
            return h.permute(1, 2, 0).reshape(B, -1)
        else:
            out = self.fc(h_all) # [B, T, CF]
            out = out.permute(0, 2, 1).reshape(B, C, F, T)
            return out

class UNetForTriplet_to1dLSTM(nn.Module):
    def __init__(self, inst_list, f_size, mono=True, to1d_mode="mean_linear", order="timefreq", mel=False, n_mels=259):
        super().__init__()
        if mono:
            encoder_in_size = 1
        else:
            encoder_in_size = 2
        encoder_out_size = len(inst_list) * 128
        # Encoder
        self.encoder = UNetEncoder(encoder_in_size, encoder_out_size)
        if mel:
            #in_channel = (n_mels//(2**6)+1)*encoder_out_size
            in_channel = math.ceil(n_mels/(2**6))*encoder_out_size
        else:
            in_channel = (f_size/2/(2**6)+1)*encoder_out_size
        self.rnn_module = nn.Sequential(
            RNNModule(in_channel, in_channel),
            RNNModule(in_channel, in_channel, last=True),
        )
        self.to1d = nn.Sequential(
            nn.Linear(in_channel*2, encoder_out_size * 2),
            nn.ReLU(),
            nn.Linear(encoder_out_size * 2, encoder_out_size)
        )
        #deviceを指定
        self.inst_list = inst_list

    def forward(self, input):
        # Encoder
        input, max, min = normalize_torch(input)
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out = self.encoder(input)

        # to1d
        rnn_out = self.rnn_module(conv6_out)
        output_emb = self.to1d(rnn_out)
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
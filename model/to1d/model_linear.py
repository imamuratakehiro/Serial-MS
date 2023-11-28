"""2次元特徴量を時間方向に平均、全結合層で1次元化。"""

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


# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass

class To1D640timefreq(nn.Module):
    def __init__(self, in_channel, to1d_mode) -> None:
        super().__init__()
        self.to1d_mode = to1d_mode
        if to1d_mode == "mean_linear":
            self.fc1 = nn.Linear(in_channel, 640)
        elif to1d_mode == "meanstd_linear":
            self.fc1 = nn.Linear(in_channel*2, 640)
        elif to1d_mode == "pool":
            self.fc1 = nn.Linear(640*2, 640)
        self.to(device)

    def forward(self, input):
        if self.to1d_mode == "mean_linear":
            out_mean = torch.mean(input, dim=3)
            out_2d = out_mean.view(out_mean.size()[0], -1)
        elif self.to1d_mode == "meanstd_linear":
            out_mean = torch.mean(input, dim=3)
            out_std = torch.std(input, dim=3)
            out_2d = torch.concat([out_mean, out_std], dim=2).view(out_mean.size()[0], -1) # [B, ch*F*2]
        elif self.to1d_mode == "pool":
            avgpool = nn.AvgPool2d(kernel_size=(input.shape[2], input.shape[3]))
            # グローバル平均プーリング
            out_mean = avgpool(input)
            maxpool = nn.MaxPool2d(kernel_size=(input.shape[2], input.shape[3]))
            # グローバル最大プーリング
            out_max = maxpool(input)
            out_2d = torch.squeeze(torch.concat([out_mean, out_max], dim=1))

        output = self.fc1(out_2d)
        return output

class To1D640Demucs(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.fc1 = nn.Linear(channel*2, 640)
    
    def forward(self, input):
        avgpool = nn.AvgPool1d(kernel_size=input.shape[2])
        # グローバル平均プーリング
        out_mean = avgpool(input)
        maxpool = nn.MaxPool1d(kernel_size=input.shape[2])
        # グローバル最大プーリング
        out_max = maxpool(input)
        out_1d = torch.squeeze(torch.concat([out_mean, out_max], dim=1))
        return self.fc1(out_1d)

class To1D128timefreq(nn.Module):
    def __init__(self, mode="mean_linear", in_channel=None, att=False) -> None:
        super().__init__()
        if mode == "mean_linear" or mode == "max_linear":
            """
            self.conv = nn.Sequential(  nn.Conv2d(128, 128, kernel_size = (5, 5), padding="same"),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(128, 128, kernel_size = (5, 5), padding="same"),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(0.2),
                                        )
            """
            self.fc1 = nn.Linear(in_channel, 128)
            if att:
                self.attention = nn.Sequential(
                    nn.Linear(in_channel, in_channel),
                    nn.GELU(),
                    nn.Linear(in_channel, in_channel),
                    nn.Sigmoid()
                )
            self.att = att
        elif mode == "avgpool":
            pass
        else:
            raise MyError(f"Argument mode is not correct ({mode}).")
        self.mode = mode
        self.to(device)

    def forward(self, input):
        if self.mode == "mean_linear" or self.mode == "max_linear":
            #input = self.conv(input)
            if self.mode == "mean_linear":
                out_mean = torch.mean(input, axis=3)
            elif self.mode == "max_linear":
                out_mean = torch.max(input, axis=3).values
            out_mean_2d = out_mean.view(out_mean.size()[0], -1)
            if self.att:
                out_mean_2d = self.attention(out_mean_2d)*out_mean_2d
            output = self.fc1(out_mean_2d)
        elif self.mode == "avgpool":
            if not "argpool" in vars(self):
                self.avgpool = nn.AvgPool2d(kernel_size=(input.shape[2], input.shape[3]))
            output = self.avgpool(input)[:,:,0,0]
        return output

class To1D128freqtime(nn.Module):
    def __init__(self, mode="mean_linear", in_channel=None) -> None:
        super().__init__()
        if mode == "mean_linear" or mode == "max_linear":
            #self.bilstm = BiLSTM_Embedding(in_channel=in_channel, out_channel=128)
            self.fc1 = nn.Linear(in_channel, 128)
        elif mode == "lstm_linear":
            self.fc1 = nn.Linear(in_channel, 128)
            self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, bidirectional=False)
        elif mode == "avgpool":
            pass
        else:
            raise MyError(f"Argument mode is not correct ({mode}).")
        self.mode = mode
        self.to(device)

    def forward(self, input):
        if self.mode in ["mean_linear", "max_linear", "lstm_linear"]:
            tmp_list = []
            for t in range(input.shape[3]):
                x = self.fc1(torch.reshape(input[:,:,:,t], (input.shape[0], -1)))
                #x = self.bilstm(torch.reshape(input[:,:,:,t], (input.shape[0], -1)))
                tmp_list.append(x)
            x = torch.stack(tmp_list, dim=2)
            if self.mode == "mean_linear":
                output = torch.mean(x, axis=2)
            elif self.mode == "max_linear":
                output = torch.max(x, axis=2).values
            elif self.mode == "lstm_linear":
                output = self.lstm(x.permute(0,2,1))[1][0][-1]
        elif self.mode == "avgpool":
            if not "argpool" in vars(self):
                self.avgpool = nn.AvgPool2d(kernel_size=(input.shape[2], input.shape[3]))
            output = self.avgpool(input)[:,:,0,0]
        return output.squeeze()

class To1dLSTM(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(int(in_channel/(128*4)), int(in_channel/(128*4)), num_layers=2, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(in_channel, 128)
    def forward(self, x):
        B, C, F, T = x.shape
        bilstm = self.bilstm(x.permute(0,1,3,2).reshape(-1, T, F))[1][0] # 系列長 = 時間、文字列長 = 周波数
        return self.fc1(bilstm[:,-1].reshape(B, -1)) # 時間方向の最後の次元のみを採用することで時間方向を1次元化

class To1dLSTM2(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(512, 512, num_layers=2, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(in_channel, 128)
    def forward(self, x):
        B, C, F, T = x.shape
        bilstm = self.bilstm(x.permute(0,2,3,1).reshape(-1, T, C))[0] # 系列長 = 時間、文字列長 = 周波数
        return self.fc1(bilstm[:,-1].reshape(B, -1)) # 時間方向の最後の次元のみを採用することで時間方向を1次元化

class To1D128freq(nn.Module):
    """周波数方向をDNNで潰して時間方向を残す"""
    def __init__(self, mode="mean_linear", in_channel=None) -> None:
        super().__init__()
        if mode == "mean_linear" or mode == "max_linear":
            self.bilstm = BiLSTM_Embedding(in_channel=128, out_channel=128)
            self.fc1 = nn.Linear(in_channel, 128)
        elif mode == "avgpool":
            pass
        else:
            raise MyError(f"Argument mode is not correct ({mode}).")
        self.mode = mode
        self.to(device)

    def forward(self, input):
        if self.mode == "mean_linear" or self.mode == "max_linear":
            tmp_list = []
            for t in range(input.shape[3]):
                x = self.fc1(torch.reshape(input[:,:,:,t], (input.shape[0], -1)))
                x = self.bilstm(x)
                tmp_list.append(x)
            output = torch.stack(tmp_list, dim=2)
            """
            if self.mode == "mean_linear":
                output = torch.mean(x, axis=2)
            elif self.mode == "max_linear":
                output = torch.max(x, axis=2).values
            """
        elif self.mode == "avgpool":
            if not "argpool" in vars(self):
                self.avgpool = nn.AvgPool2d(kernel_size=(input.shape[2], input.shape[3]))
            output = self.avgpool(input)[:,:,0,0]
        return output

class To2DFrom1DFreqTime(nn.Module):
    def __init__(self, out_channel) -> None:
        super().__init__()
        self.linear = nn.Linear(640, out_channel)

    def forward(self, input, time_dim):
        B, _ = input.shape
        x = self.linear(input)
        x = torch.reshape(x, (B, 640, -1, 1))
        out = []
        for t in range(time_dim):
            out.append(x.clone())
        return torch.concat(out, dim=3)


class To1D128(nn.Module):
    def __init__(self, to1d_mode, order, in_channel, att=False) -> None:
        super().__init__()
        if order == "timefreq":
            self.to1d = To1D128timefreq(mode=to1d_mode, in_channel=int(in_channel), att=att)
        elif order =="freqtime":
            self.to1d = To1D128freqtime(mode=to1d_mode, in_channel=int(in_channel))
        elif order == "freq_emb_time":
            pass
            #self.embnet = To1DFreqEmbedding(to1d_mode=to1d_mode, in_channel_freq=int(in_channel), tanh=tanh)
        elif order == "bilstm":
            self.to1d = To1dLSTM2(in_channel=int(in_channel))
        self.recog = nn.Linear(128, 1)
        #deviceを指定
        self.to(device)

    def forward(self, input):
        emb_vec = self.to1d(input)
        recog_p = self.recog(emb_vec)
        return emb_vec, recog_p

class To1D640(nn.Module):
    def __init__(self, to1d_mode, order, in_channel) -> None:
        super().__init__()
        if order == "timefreq":
            self.to1d = To1D640timefreq(in_channel=int(in_channel), to1d_mode=to1d_mode)
        elif order =="freqtime":
            pass
        elif order == "freq_emb_time":
            pass
            #self.embnet = To1DFreqEmbedding(to1d_mode=to1d_mode, in_channel_freq=int(in_channel), tanh=tanh)
        elif order == "bilstm":
            pass
        #deviceを指定
        self.to(device)

    def forward(self, input):
        emb_vec = self.to1d(input)
        return emb_vec

class To2DFrom1D640(nn.Module):
    def __init__(self, out_channel) -> None:
        super().__init__()
        self.to2d = To2DFrom1DFreqTime(out_channel=int(out_channel))
    
    def forward(self, input, time_dim):
        x = self.to2d(input, time_dim)
        return x


class BiLSTM_Embedding(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(in_channel, out_channel, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(out_channel*2, out_channel)
    
    def forward(self, input):
        out, hc = self.bilstm(input)
        output = self.linear(out)
        return output

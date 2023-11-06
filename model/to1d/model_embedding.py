"""エンコーダ出力をembeddingするネットワーク"""

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
from ..csn import ConditionalSimNet1d
from .model_linear import To1D128freqtime, To1D128timefreq, To1D128freq, BiLSTM_Embedding


# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass

class EmbeddingNet128(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(640, 128),
                                nn.LeakyReLU(0.2),
                                nn.Linear(128, 128),
                                nn.Tanh())
    
    def forward(self, input):
        output = self.fc(input)
        return output

class EmbeddingNet128to128(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(128, 128),
                                nn.LeakyReLU(0.2),
                                nn.Linear(128, 128),)
    
    def forward(self, input):
        output = self.fc(input)
        return output

class EmbeddingNet640to640(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(640, 640),
                                nn.LeakyReLU(0.2),
                                nn.Linear(640, 640),)
    
    def forward(self, input):
        output = self.fc(input)
        return output

class EmbeddingNet(nn.Module):
    def __init__(self, inst_list, to1d_mode, order, f_size, tanh) -> None:
        super().__init__()
        for inst in inst_list:
            if inst == "drums":
                self.embnet_drums  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*128, tanh=tanh)
            elif inst == "bass":
                self.embnet_bass  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*128, tanh=tanh)
            elif inst == "piano":
                self.embnet_piano  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*128, tanh=tanh)
            elif inst == "guitar":
                self.embnet_guitar  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*128, tanh=tanh)
            elif inst == "vocals":
                self.embnet_vocals  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*128, tanh=tanh)
            elif inst == "residuals":
                self.embnet_residuals  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*128, tanh=tanh)
            elif inst == "other":
                self.embnet_other  = To1dEmbedding(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*128, tanh=tanh)
        self.inst_list = inst_list
        self.sigmoid = nn.Sigmoid()
        self.to(device)
    
    def forward(self, input):
        embnet = {}
        output_emb = {}
        output_probability = {}
        for inst in self.inst_list:
            if inst == "drums":
                embnet[inst] = self.embnet_drums
            elif inst == "bass":
                embnet[inst] = self.embnet_bass
            elif inst == "piano":
                embnet[inst] = self.embnet_piano
            elif inst == "guitar":
                embnet[inst] = self.embnet_guitar
            elif inst == "vocals":
                embnet[inst] = self.embnet_vocals
            elif inst == "residuals":
                embnet[inst] = self.embnet_residuals
            elif inst == "other":
                embnet[inst] = self.embnet_other
        for idx, inst in enumerate(self.inst_list):
            # embedding
            emb, recog_probability = embnet[inst](input[:,128*idx : 128*(idx+1), :,:])
            output_emb[inst] = emb
            # 原点からのユークリッド距離にtanhをしたものを無音有音の確率とする
            #output_probability[inst] = self.sigmoid(torch.log(torch.sqrt(torch.sum(emb**2, dim=1))))
            output_probability[inst] = self.sigmoid(recog_probability)[:,0]
            #print(output_probability[inst].shape)
        return output_emb, output_probability

class To1DFreqEmbedding(nn.Module):
    def __init__(self, to1d_mode, in_channel_freq, tanh = True) -> None:
        super().__init__()
        if to1d_mode == "mean_linear" or to1d_mode == "max_linear":
            self.fc1 = nn.Linear(in_channel_freq, 128)
            self.bilstm = BiLSTM_Embedding(in_channel=1, out_channel=1)
            self.emb = EmbeddingNet128to128()
        else:
            raise MyError(f"Argument mode is not correct ({to1d_mode}).")
        self.to1d_mode = to1d_mode
        self.recog = nn.Linear(128, 1)
        self.to(device)
        
    def forward(self, input):
        tmp_list = []
        for t in range(input.shape[3]):
            x = self.fc1(torch.reshape(input[:,:,:,t], (input.shape[0], -1)))
            x = x.unsqueeze(dim=2)
            #print(x.shape)
            x = self.bilstm(x)
            x = torch.squeeze(x)
            #print(x.shape)
            x = self.emb(x)
            tmp_list.append(x)
        emb_time = torch.stack(tmp_list, dim=2)
        #emb_vec = torch.mean(emb_time, dim=2)
        if self.to1d_mode == "mean_linear":
            emb_vec = torch.mean(emb_time, axis=2)
        elif self.to1d_mode == "max_linear":
            emb_vec = torch.max(emb_time, axis=2).values
        return emb_vec

class To1dEmbedding(nn.Module):
    def __init__(self, cfg, in_channel) -> None:
        super().__init__()
        if cfg.order == "timefreq":
            self.embnet = nn.Sequential(To1D128timefreq(mode=cfg.to1d_mode, in_channel=int(in_channel)),
                                    #BiLSTM_Embedding(in_channel=128, out_channel=128),
                                    #EmbeddingNet128to128(tanh)
            )
        elif cfg.order =="freqtime":
            self.embnet = nn.Sequential(To1D128freqtime(mode=cfg.to1d_mode, in_channel=int(in_channel)),
                                    #BiLSTM_Embedding(in_channel=128, out_channel=128),
                                    #EmbeddingNet128to128(tanh)
            )
        elif cfg.order == "freq_emb_time":
            self.embnet = To1DFreqEmbedding(to1d_mode=cfg.to1d_mode, in_channel_freq=int(in_channel))
        if cfg.order in ["timefreq", "freqtime"] and cfg.embnet:
            self.embnet.add_module("embnet", EmbeddingNet128to128())
        self.recog = nn.Linear(128, 1)
        #deviceを指定
        self.to(device)

    def forward(self, input):
        emb_vec = self.embnet(input)
        recog_p = self.recog(emb_vec)
        return emb_vec, recog_p

        


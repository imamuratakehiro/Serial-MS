import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
#from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torchvision.transforms as Tv
import numpy as np
import os
#import stempeg
import csv
import pandas as pd
import soundfile as sf
#from tqdm_table import tqdm_table
#from tqdm import tqdm
import json
import librosa.core as lc
import librosa
import random

from utils.func import stft, trackname, detrackname

class MyError(Exception):
    pass

def SongDataFile(cfg, mode):
    # datasetのmodeの確認
    if not mode in ["train", "valid", "test"]:
        raise MyError(f"Argument type is not correct ({cfg.mode}).")
    # データファイルの読み込み
    if mode == "train":
        seconds = cfg.seconds_train
    else:
        seconds = cfg.seconds_test
    if cfg.datasetname == "slakh":
        path = "/nas03/assets/Dataset/slakh-2100_2"
        if cfg.reduce_silence=="mix" or cfg.reduce_silence=="stems":
            datafile = np.loadtxt(f"./metadata/{cfg.datasetname}/{seconds}s_no_silence_or0.5_0.25/{mode}_slakh_{seconds}s_{cfg.reduce_silence}.txt", delimiter = ",", dtype = "unicode")
        elif cfg.reduce_silence=="none":
            datafile = np.loadtxt(f"./metadata/{cfg.datasetname}/{seconds}s/{mode}_slakh_{seconds}s.txt", delimiter = ",", dtype = "unicode")
        else:
            raise MyError(f"Argument reduce_silence is not correct ({cfg.reduce_silence}).")
    elif cfg.datasetname == "musdb18":
        path = "/nas03/assets/Dataset/MUSDB18/wav"
        if cfg.reduce_silence=="mix" or cfg.reduce_silence=="stems":
            datafile = np.loadtxt(f"./metadata/{cfg.datasetname}/{seconds}s_no_silence_or0.5_0.25/{mode}_{cfg.datasetname}_{seconds}s_{cfg.reduce_silence}.txt", delimiter = ",", dtype = "unicode")
        elif cfg.reduce_silence=="none":
            datafile = np.loadtxt(f"./metadata/{cfg.datasetname}/{seconds}s/train_{cfg.datasetname}_{seconds}s.txt", delimiter = ",", dtype = "unicode")
        else:
            raise MyError(f"Argument reduce_silence is not correct ({cfg.reduce_silence}).")
    else:
        raise MyError(f"Argument datasetname is not correct ({cfg.datasetname}).")
    return datafile, path

class LoadSeg(Dataset):
    def __init__(
        self,
        cfg,
        inst:str,
        mode: str,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.inst = inst
        self.datafile, self.path = SongDataFile(cfg, mode=mode)
    
    def __len__(self):
        return self.datafile.shape[0]
    
    def __getitem__(self, idx):
        if self.cfg.datasetname == "slakh":
            stem_path = self.path + "/" + self.datafile[idx][0] + f"/submixes/{self.inst}.wav"
        elif self.cfg.datasetname == "musdb18":
            stem_path = self.path + f"/{self.mode}/" + self.datafile[idx][0] + f"/{self.inst}.wav"
        n = int(self.datafile[idx][1])
        sound_n, sr = librosa.load(stem_path,
                            sr=None,
                            mono=self.cfg.mono,
                            offset=n/self.cfg.sr,
                            duration=self.cfg.seconds_train if self.mode == "train" else self.cfg.seconds_test)
        if self.cfg.mono:
            sound_n = np.array([sound_n]) # 次元を追加
        return torch.from_numpy(sound_n)

class SongData(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        mode="train",
        cases = 31,
        ) -> None:
        super().__init__()
        # datasetのmodeの確認
        self.cfg = cfg
        self.mode = mode
        self.cases = cases
        self.datafile, self.path = SongDataFile(cfg, mode=mode)
        self.loadseg = {}
        for inst in cfg.inst_list:
            self.loadseg[inst] = LoadSeg(cfg, inst, mode=mode)

    def __len__(self):
        return self.datafile.shape[0]

    def __getitem__(self, idx):
        bin_str = format(self.cases, f"0{len(self.inst_list)}b") #2進数化
        if self.mono:
            mix_wave = torch.zeros(1, self.seconds*self.sr)
        else:
            mix_wave = torch.zeros(2, self.seconds*self.sr)
        #stem_transformed_list = []
        stem_wave_list = []
        # 2進数化された条件からmix音源・stem音源を生成
        for j, inst in enumerate(self.inst_list):
            if bin_str[j] == "1":
                sound_t_stem = self.loadseg[inst][idx]
                mix_wave += sound_t_stem
            else:
                sound_t_stem =  torch.zeros_like(mix_wave)
            stem_wave_list.append(sound_t_stem)
        stem_wave = torch.stack(stem_wave_list, axis=0)
        mix_param, mix_transformed,  _ = stft(mix_wave, f_size=self.f_size)
        # stemはmixと同じparamでnormalize
        _, stem_transformed, _ = stft(stem_wave, f_size=self.f_size, param=mix_param)
        return mix_transformed, stem_transformed





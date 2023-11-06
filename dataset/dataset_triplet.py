from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
#from torchvision.transforms import ToTensor, Lambda
from typing import Any, Dict, Optional, Tuple
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
from utils.logger import MyLoggerModel, MyLoggerTrain
from .dataset_slakh_musdb18 import LoadSeg, SongDataFile


class LoadSongWithLabel(Dataset):
    def __init__(
        self,
        cfg,
        mode) -> None:
        super().__init__()
        self.cfg = cfg
        self.datafile, self.path = SongDataFile(cfg, mode=mode)
        self.loadseg = {inst: LoadSeg(self.cfg, inst, mode=mode) for inst in self.cfg.inst_list}
    
    def __len__(self):
        return self.datafile.shape[0]
    
    def __getitem__(self, idx) -> Any:
        label = detrackname(self.datafile[idx][0])
        song_wave = sum([self.loadseg[inst][idx] for inst in self.cfg.inst_list])
        _, song_transformed, _ = stft(song_wave, cfg=self.cfg)
        return torch.tensor(label), song_transformed


class TripletDatasetOneInst(Dataset):
    def __init__(
        self,
        mode: str,
        cfg) -> None:
        super().__init__()
        self.cfg = cfg
        if mode == "train":
            self.triplet = self.make_tripletfile(mode="train", n_triplet=self.cfg.n_triplet_train)
            self.loadseg = {inst: LoadSeg(self.cfg, inst, mode="train") for inst in self.cfg.inst_list}
        elif mode == "valid":
            self.triplet = self.make_tripletfile(mode="valid", n_triplet=self.cfg.n_triplet_valid)
            self.loadseg = {inst: LoadSeg(self.cfg, inst, mode="valid") for inst in self.cfg.inst_list}
        elif mode == "test":
            self.triplet = self.make_tripletfile(mode="test", n_triplet=self.cfg.n_triplet_test)
            self.loadseg = {inst: LoadSeg(self.cfg, inst, mode="test") for inst in self.cfg.inst_list}
        elif mode == "predict":
            pass
        
    def make_tripletfile(self, mode: str, n_triplet):
        # tripletを作成
        datafile, _ = SongDataFile(self.cfg, mode=mode)
        log = [] # 既に出したセグメント記録用
        triplet = []
        counter = 0
        while counter < n_triplet:
            # anchor
            anchor = random.randint(0, len(datafile) - 1)
            while anchor in log:
                anchor = random.randint(0, len(datafile) - 1)
            # negative
            negative = random.randint(0, len(datafile) - 1)
            while (negative in log or
                    datafile[negative][0] == datafile[anchor][0]): # 同じ曲から取り出されない
                negative = random.randint(0, len(datafile) - 1)
            # positive
            songseglist = np.where(datafile[:,0] == datafile[anchor][0])[0].tolist() #なんか2重になって出てくるから、1重に
            songseglist.remove(anchor)
            if len(songseglist) == 0:
                continue
            positive = random.choice(songseglist)
            log.append(anchor)
            log.append(negative)
            #log.append(positive)
            triposi = [0]; posiposi = [1]
            triplet.append([anchor, positive, negative, triposi, posiposi])
            counter += 1
        return triplet
    
    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, idx) -> None:
        anchor, positive, negative, triposi, posiposi = self.triplet[idx]
        for inst in self.cfg.inst_list:
            _, a_X, _ = stft(self.loadseg[inst][anchor],   cfg=self.cfg)
            _, p_X, _ = stft(self.loadseg[inst][positive], cfg=self.cfg)
            _, n_X, _ = stft(self.loadseg[inst][negative], cfg=self.cfg)
        return a_X, p_X, n_X, torch.tensor(triposi), torch.tensor(posiposi)
    

class TripletDatasetBA(Dataset):
    def __init__(
        self,
        cfg,
        mode="train",
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.loadseg = {}
        for inst in cfg.inst_list:
            self.loadseg[inst] = LoadSeg(cfg, inst, mode=mode)
        if mode == "train":
            self.sametempolist = json.load(open("./metadata/slakh/sametempo_train_edited.json", 'r'))
        elif mode == "test":
            self.sametempolist = json.load(open("./metadata/slakh/sametempo_test_edited.json", 'r'))
        
        self.log = [] # 既に出したセグメント記録用
    
    def get_triplet_pse(self):
        #for i in range(self.cfg.)
        while True:
            fail = False
            # anchor
            anchor = random.randint(0, self.len - 1)
            while anchor in self.log:
                anchor = random.randint(0, self.len - 1)
            # positive
            # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
            if self.datafile[anchor+2][0] == self.datafile[anchor][0]:
                positive = anchor+2
            elif self.datafile[anchor-2][0] == self.datafile[anchor][0]:
                positive = anchor-2
            else:
                fail=True
            if fail:
                print(f"\There is no same song seg to anchor : {self.datafile[anchor][0]}")
                self.log.append(anchor)
                continue
            if self.track_exist_checker_in_additional(anchor):
                print(f"\tanchor not found fail : {self.datafile[anchor][0]}") # sametempolistに曲がない時
                self.log.append(anchor)
                continue
            # negative
            # 同じテンポでanchorとは違う曲をnegativeとして2曲取り出す
            for stlist in self.sametempolist:
                if detrackname(self.datafile[anchor][0]) in stlist:
                    negative_songs = random.sample(stlist, 2)
                    while detrackname(self.datafile[anchor][0]) in negative_songs:
                        negative_songs = random.sample(stlist, 2)
                    # negativeの曲から、片方から1セグメント、もう片方から隣り合う2セグメント取り出す
                    negative_song1 = negative_songs[0]; negative_song2 = negative_songs[1]
                    n_seg_list1 = np.where(self.datafile[:,0] == trackname(negative_song1))[0]
                    n_seg_list2 = np.where(self.datafile[:,0] == trackname(negative_song2))[0]
                    if len(n_seg_list1) == 0 or len(n_seg_list2) == 0:
                        fail = True # 無音排除で曲丸ごとない時
                        break
                    negative = random.choice(n_seg_list1)
                    new_a = random.choice(n_seg_list2[:-1])
                    new_p = new_a + 1
                    if self.datafile[new_a][0] != self.datafile[new_p][0]:
                        fail = True
                    #counter = 0
                    #while negative in self.log:
                    #    if counter > 10:
                    #        fail = True; break
                    #    negative = random.choice(n_seg_list)
                    #    counter += 1
                    break
            if fail:
                print(f"\tnegative not found fail : {trackname(negative_song1)}, {trackname(negative_song2)}")
                continue
            break
        self.log.append(anchor)
        self.log.append(negative)
        self.log.append(positive)
        

    def __len__(self):
        return len(self.datafile)//3

    def track_exist_checker_in_additional(self, track_id):
        for stlist in self.sametempolist:
            if detrackname(self.datafile[track_id][0]) in stlist:
                return True
        return False
    
    def export_cases_from_triplet_inst(self, triplet_inst):
        inst_index = self.inst_list.index(triplet_inst)
        max = 2**len(self.inst_list)-1
        index = 2**(len(self.inst_list)-1-inst_index)
        return index, max - index
    
    def __getitem__(self, index) -> None:
        while True:
            fail = False
            # anchor
            anchor = random.randint(0, self.len - 1)
            while anchor in self.log:
                anchor = random.randint(0, self.len - 1)
            # positive
            # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
            if self.datafile[anchor+2][0] == self.datafile[anchor][0]:
                positive = anchor+2
            elif self.datafile[anchor-2][0] == self.datafile[anchor][0]:
                positive = anchor-2
            else:
                fail=True
            if fail:
                print(f"\There is no same song seg to anchor : {self.datafile[anchor][0]}")
                self.log.append(anchor)
                continue
            if self.track_exist_checker_in_additional(anchor):
                print(f"\tanchor not found fail : {self.datafile[anchor][0]}") # sametempolistに曲がない時
                self.log.append(anchor)
                continue
            # negative
            # 同じテンポでanchorとは違う曲をnegativeとして2曲取り出す
            for stlist in self.sametempolist:
                if detrackname(self.datafile[anchor][0]) in stlist:
                    negative_songs = random.sample(stlist, 2)
                    while detrackname(self.datafile[anchor][0]) in negative_songs:
                        negative_songs = random.sample(stlist, 2)
                    # negativeの曲から、片方から1セグメント、もう片方から隣り合う2セグメント取り出す
                    negative_song1 = negative_songs[0]; negative_song2 = negative_songs[1]
                    n_seg_list1 = np.where(self.datafile[:,0] == trackname(negative_song1))[0]
                    n_seg_list2 = np.where(self.datafile[:,0] == trackname(negative_song2))[0]
                    if len(n_seg_list1) == 0 or len(n_seg_list2) == 0:
                        fail = True # 無音排除で曲丸ごとない時
                        break
                    negative = random.choice(n_seg_list1)
                    new_a = random.choice(n_seg_list2[:-1])
                    new_p = new_a + 1
                    if self.datafile[new_a][0] != self.datafile[new_p][0]:
                        fail = True
                    #counter = 0
                    #while negative in self.log:
                    #    if counter > 10:
                    #        fail = True; break
                    #    negative = random.choice(n_seg_list)
                    #    counter += 1
                    break
            if fail:
                print(f"\tnegative not found fail : {trackname(negative_song1)}, {trackname(negative_song2)}")
                continue
            break
        self.log.append(anchor)
        self.log.append(negative)
        self.log.append(positive)
        # anchor、positive、negativeの順で出力
        posiposi = [1, 1, 1, 1, 1]
        negaposi = [2, 2, 2, 2, 2]
        #print(anchor, positive, negative)
        triplet_index = random.choice(list(range(len(self.inst_list))))
        #cases_triplet, cases_not_triplet = self.export_cases_from_triplet_inst(triplet_inst)
        #self.dataset.cases = cases_not_triplet
        _, a_y = self.dataset[anchor]; _, p_y = self.dataset[positive]; _, n_y = self.dataset[negative]
        #self.dataset.cases = cases_triplet
        _, na_y = self.dataset[new_a]; _, np_y = self.dataset[new_p]
        a_y[triplet_index], na_y[triplet_index] = na_y[triplet_index], a_y[triplet_index]
        p_y[triplet_index], n_y[triplet_index]  = n_y[triplet_index],  p_y[triplet_index]
        n_y[triplet_index], np_y[triplet_index] = np_y[triplet_index], n_y[triplet_index]
        posiposi[triplet_index], negaposi[triplet_index] = negaposi[triplet_index], posiposi[triplet_index]
        a_X = torch.sum(a_y, axis=0); p_X = torch.sum(p_y, axis=0); n_X = torch.sum(n_y, axis=0)
        # 波形をスペクトログラムに
        a_X_s = wave2spectro_1d(a_X, self.f_size); p_X_s = wave2spectro_1d(p_X, self.f_size); n_X_s = wave2spectro_1d(n_X, self.f_size)
        a_y_s = wave2spectro_2d(a_y, self.f_size); p_y_s = wave2spectro_2d(p_y,  self.f_size); n_y_s = wave2spectro_2d(n_y,  self.f_size)
        return [a_X_s, a_y_s], [p_X_s, p_y_s], [n_X_s, n_y_s], torch.tensor(triplet_index), torch.tensor(posiposi)
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
from omegaconf import OmegaConf,open_dict

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
            self.sametempolist = json.load(open(cfg.metadata_dir + "slakh/sametempo_train_edited.json", 'r'))
        elif mode == "test":
            self.sametempolist = json.load(open(cfg.metadata_dir + "slakh/sametempo_test_edited.json", 'r'))
        
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
    
class BA4Track:
    def __init__(self, n_triplet, cfg, mode) -> None:
        self.n_triplet = n_triplet
        self.cfg = cfg
        self.n_inst = len(cfg.inst_list)
        self.sametempolist = json.load(open(cfg.metadata_dir + f"slakh/sametempo_{mode}_edited.json", 'r'))
        self.songlist_in_sametempolist = np.loadtxt(cfg.metadata_dir + f"slakh/songlist_in_sametempo_list_{mode}.txt", delimiter = ",", dtype = "int64")
        self.datafile = pd.read_csv(cfg.metadata_dir + f"zume/slakh/3s_on1.5_in_sametempolist_{mode}.csv").values

    def __len__(self):
        return self.n_triplet
    
    def pick_seg_from_track(self, track):
        segs = np.where(self.target[:,0] == int(track))[0]
        if len(segs) == 0:
            return -1
        seg = random.choice(segs)
        return seg
    
    def pick_p_from_a(self, anchor):
        # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
        if anchor+2 >= len(self.target):
            return -1
        if self.target[anchor+2][0] == self.target[anchor][0]:
            #positive = anchor+2
            return anchor+2
        elif self.target[anchor-2][0] == self.target[anchor][0]:
            #positive = anchor-2
            return anchor-2
        else:
            #fail=True
            return -1
    
    def make_pseudo(self, triposi, a, p, n, a2, p2, n2):
        #print(a, p, n, a2, p2, n2)
        track_a = [self.target[a][0] for i in range(self.n_inst)]; track_a[triposi] = self.target[a2][0]
        track_p = [self.target[n][0] for i in range(self.n_inst)]; track_p[triposi] = self.target[p2][0]
        track_n = [self.target[p][0] for i in range(self.n_inst)]; track_n[triposi] = self.target[n2][0]
        seg_a   = [self.target[a][1] for i in range(self.n_inst)]; seg_a[triposi] = self.target[a2][1]
        seg_p   = [self.target[n][1] for i in range(self.n_inst)]; seg_p[triposi] = self.target[p2][1]
        seg_n   = [self.target[p][1] for i in range(self.n_inst)]; seg_n[triposi] = self.target[n2][1]
        sound_a = [self.target[a][i + 2] for i in range(self.n_inst)]; sound_a[triposi] = self.target[a2][triposi + 2]
        sound_p = [self.target[n][i + 2] for i in range(self.n_inst)]; sound_p[triposi] = self.target[p2][triposi + 2]
        sound_n = [self.target[p][i + 2] for i in range(self.n_inst)]; sound_n[triposi] = self.target[n2][triposi + 2]
        return track_a, track_p, track_n, seg_a, seg_p, seg_n, sound_a, sound_p, sound_n

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """ba_4trackの楽曲を生成する関数"""
        b, a = random.sample(range(0, len(self.cfg.inst_list)), 2) # basicとadditionalの場所を決定
        # basic, additional共に有音のセグのみを選択対象に
        self.target = self.datafile[np.where((self.datafile[:, b + 2] == 1) & (self.datafile[:, a + 2] == 1))[0]]
        while True:
            fail = False
            # anchor
            a_track = random.choice(self.songlist_in_sametempolist)
            a_idx = self.pick_seg_from_track(a_track)
            if a_idx == -1:
                #print(f"\tAnchor does not have any segs. : {a_track}")
                continue
            # positive
            # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
            p_idx = self.pick_p_from_a(a_idx)
            if fail or p_idx == -1:
                #print(f"\tThere is no same song seg to anchor : {self.target[a_idx][0]}")
                #log.append(anchor)
                continue
            # negative
            # 同じテンポでanchorとは違う曲をnegativeとして3曲取り出す
            found = False
            for sametempo in self.sametempolist:
                if int(a_track) in sametempo:
                    #print(type(int(a_track)))
                    # negativeの曲から、2つから1セグメント、もう1つから隣り合う2セグメント取り出す
                    negative_songs = random.sample(sametempo, 3)
                    while int(a_track) in negative_songs: # anchorと曲が被っていたらやり直し
                        negative_songs = random.sample(sametempo, 3)
                    n_track, a2_track, n2_track = negative_songs
                    n_idx  = self.pick_seg_from_track(n_track)
                    a2_idx = self.pick_seg_from_track(a2_track)
                    n2_idx = self.pick_seg_from_track(n2_track)
                    if n_idx == -1 or a2_idx == -1 or n2_idx == -1:
                        fail = True # 無音排除で曲丸ごとない時
                        break
                    p2_idx = self.pick_p_from_a(a2_idx)
                    if p2_idx == -1:
                        fail = True
                    found = True
                    break
            if fail:
                #print(f"\tnegative not found fail : {trackname(n_track)}, {trackname(a2_track)}, {trackname(n2_track)}")
                continue
            if not found:
                print(a_track)
            break
        track_a, track_p, track_n, seg_a, seg_p, seg_n, sound_a, sound_p, sound_n = self.make_pseudo(b, a_idx, p_idx, n_idx, a2_idx, p2_idx, n2_idx)
        return track_a, track_p, track_n, seg_a, seg_p, seg_n, sound_a, sound_p, sound_n, [b, a]
        #return track_a, track_n, track_p, seg_a, seg_n, seg_p, sound_a, sound_p, sound_n, a

class BA4Track_31ways:
    def __init__(self, n_triplet, cfg, mode) -> None:
        self.sametempolist = json.load(open(cfg.metadata_dir + f"slakh/sametempo_{mode}_edited.json", 'r'))
        self.datafile = np.loadtxt(cfg.metadata_dir + f"slakh/3s_no_silence_or0.5_0.25/{mode}_slakh_3s_stems.txt", delimiter = ",", dtype = "unicode")
        self.datafile = self.offset2num(self.datafile, offset=3/2)
        self.songlist_in_sametempolist = np.loadtxt(cfg.metadata_dir + f"slakh/songlist_in_sametempo_list_{mode}.txt", delimiter = ",", dtype = "unicode")
        self.n_triplet = n_triplet
        self.cfg = cfg
        self.n_inst = len(cfg.inst_list)
        #self.seconds = seconds
        self.mode = mode
    
    def __len__(self):
        return self.n_triplet
    
    def offset2num(self, file, offset):
        for idx, seg in enumerate(file):
            num = int(seg[1])/44100/offset
            file[idx][1] = int(num)
        return file
    
    def pick_seg_from_track(self, track):
        segs = np.where(self.datafile[:,0] == trackname(int(track)))[0]
        if len(segs) == 0:
            return -1
        seg = random.choice(segs)
        return seg

    def track_exist_checker_in_additional(self, track_id):
        for stlist in self.sametempolist:
            if detrackname(self.datafile[track_id][0]) in stlist:
                return True
        return False
    
    def pick_p_from_a(self, anchor):
        # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
        if anchor+2 >= len(self.datafile):
            return -1
        if self.datafile[anchor+2][0] == self.datafile[anchor][0]:
            #positive = anchor+2
            return anchor+2
        elif self.datafile[anchor-2][0] == self.datafile[anchor][0]:
            #positive = anchor-2
            return anchor-2
        else:
            #fail=True
            return -1
    
    def make_pseudo(self, triposi, sound, a, p, n, a2, p2, n2):
        #print(a, p, n, a2, p2, n2)
        track_a = [detrackname(self.datafile[a][0]) if sound[i] == 1 else -1 for i in range(self.n_inst)]; track_a[triposi] = detrackname(self.datafile[a2][0])
        track_p = [detrackname(self.datafile[n][0]) if sound[i] == 1 else -1 for i in range(self.n_inst)]; track_p[triposi] = detrackname(self.datafile[p2][0])
        track_n = [detrackname(self.datafile[p][0]) if sound[i] == 1 else -1 for i in range(self.n_inst)]; track_n[triposi] = detrackname(self.datafile[n2][0])
        seg_a   = [self.datafile[a][1] if sound[i] == 1 else -1 for i in range(self.n_inst)]; seg_a[triposi] = self.datafile[a2][1]
        seg_p   = [self.datafile[n][1] if sound[i] == 1 else -1 for i in range(self.n_inst)]; seg_p[triposi] = self.datafile[p2][1]
        seg_n   = [self.datafile[p][1] if sound[i] == 1 else -1 for i in range(self.n_inst)]; seg_n[triposi] = self.datafile[n2][1]
        return track_a, track_p, track_n, seg_a, seg_p, seg_n
    
    def adjust_format(self, triplet):
        transformed = []
        for t in triplet:
            tmp = [", ".join([str(x) for x in i]) for i in t] # listの中身をstrに変換してからlist->str変換
            transformed.append("; ".join(tmp))
        return transformed


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """ba_4trackの楽曲で、有音無音全通り(31通り)の楽曲を生成する関数"""
        while True:
            fail = False
            # anchor
            a_song = random.choice(self.songlist_in_sametempolist)
            anchor = self.pick_seg_from_track(a_song)
            if anchor == -1:
                #print(f"\tAnchor does not have any segs. : {a_song}")
                continue
            # positive
            # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
            positive = self.pick_p_from_a(anchor)
            if fail or positive == -1:
                #print(f"\tThere is no same song seg to anchor : {self.datafile[anchor][0]}")
                #log.append(anchor)
                continue
            #if not self.track_exist_checker_in_additional(anchor):
            #    print(f"\tanchor not found fail : {self.datafile[anchor]}") # sametempolistに曲がない時
            #    self.log.append(anchor)
            #    continue
            # negative
            # 同じテンポでanchorとは違う曲をnegativeとして3曲取り出す
            for stlist in self.sametempolist:
                if detrackname(self.datafile[anchor][0]) in stlist:
                    # negativeの曲から、2つから1セグメント、もう1つから隣り合う2セグメント取り出す
                    negative_songs = random.sample(stlist, 3)
                    while int(a_song) in negative_songs: # anchorと曲が被っていたらやり直し
                        negative_songs = random.sample(stlist, 3)
                    n1 = negative_songs[0]; n2 = negative_songs[1]; n3 = negative_songs[2]
                    negative  = self.pick_seg_from_track(n1)
                    anchor2   = self.pick_seg_from_track(n2)
                    negative2 = self.pick_seg_from_track(n3)
                    if negative == -1 or anchor2 == -1 or negative2 == -1:
                        fail = True # 無音排除で曲丸ごとない時
                        break
                    positive2 = self.pick_p_from_a(anchor2)
                    if positive2 == -1:
                        fail = True
                    break
            if fail:
                #print(f"\tnegative not found fail : {trackname(n1)}, {trackname(n2)}, {trackname(n3)}")
                continue
            break
        c = random.randint(1, 31) # 0(完全無音)はtripletできないので省く
        c = format(c, f"0{self.n_inst}b") #2進数化
        sound = [int(i) for i in c]
        c = np.array(sound)
        # basic,additionalの場所を決めて保存
        place1 = np.where(c==1)[0]
        #print(place1)
        if len(place1) == 1:
            b = place1[0]
            #print(b)
            track_a, track_p, track_n, seg_a, seg_p, seg_n = self.make_pseudo(b, sound, anchor, positive, negative, anchor2, positive2, negative2)
            return track_a, track_p, track_n, seg_a, seg_p, seg_n, sound, sound, sound, [b, -1]
        else:
            b, a = np.random.choice(place1, 2, replace=False)
            #print(b, a)
            track_a, track_p, track_n, seg_a, seg_p, seg_n = self.make_pseudo(b, sound, anchor, positive, negative, anchor2, positive2, negative2)
            return track_a, track_p, track_n, seg_a, seg_p, seg_n, sound, sound, sound, [b, a]


class B4TrackInst:
    def __init__(self, n_triplet, cfg, mode) -> None:
        self.n_triplet = n_triplet
        self.cfg = cfg
        self.n_inst = len(["drums", "bass", "piano", "guitar", "residuals"])
        self.sametempolist = json.load(open(cfg.metadata_dir + f"slakh/sametempo_{mode}_edited.json", 'r'))
        self.songlist_in_sametempolist = np.loadtxt(cfg.metadata_dir + f"slakh/songlist_in_sametempo_list_{mode}.txt", delimiter = ",", dtype = "int64")
        self.datafile = pd.read_csv(cfg.metadata_dir + f"zume/slakh/3s_on1.5_in_sametempolist_{mode}.csv").values

    def __len__(self):
        return self.n_triplet
    
    def pick_seg_from_track(self, track):
        segs = np.where(self.target[:,0] == int(track))[0]
        if len(segs) == 0:
            return -1
        seg = random.choice(segs)
        return seg
    
    def pick_p_from_a(self, anchor):
        # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
        if anchor+2 >= len(self.target):
            return -1
        if self.target[anchor+2][0] == self.target[anchor][0]:
            #positive = anchor+2
            return anchor+2
        elif self.target[anchor-2][0] == self.target[anchor][0]:
            #positive = anchor-2
            return anchor-2
        else:
            #fail=True
            return -1
    
    def make_pseudo(self, triposi, a, p, n, a2, p2, n2):
        #print(a, p, n, a2, p2, n2)
        track_a = [self.target[a][0] for i in range(self.n_inst)]; track_a[triposi] = self.target[a2][0]
        track_p = [self.target[n][0] for i in range(self.n_inst)]; track_p[triposi] = self.target[p2][0]
        track_n = [self.target[p][0] for i in range(self.n_inst)]; track_n[triposi] = self.target[n2][0]
        seg_a   = [self.target[a][1] for i in range(self.n_inst)]; seg_a[triposi] = self.target[a2][1]
        seg_p   = [self.target[n][1] for i in range(self.n_inst)]; seg_p[triposi] = self.target[p2][1]
        seg_n   = [self.target[p][1] for i in range(self.n_inst)]; seg_n[triposi] = self.target[n2][1]
        sound_a = [self.target[a][i + 2] for i in range(self.n_inst)]; sound_a[triposi] = self.target[a2][triposi + 2]
        sound_p = [self.target[n][i + 2] for i in range(self.n_inst)]; sound_p[triposi] = self.target[p2][triposi + 2]
        sound_n = [self.target[p][i + 2] for i in range(self.n_inst)]; sound_n[triposi] = self.target[n2][triposi + 2]
        return track_a, track_p, track_n, seg_a, seg_p, seg_n, sound_a, sound_p, sound_n

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """ba_4trackの楽曲を生成する関数"""
        b = self.cfg.inst_all.index(self.cfg.inst) # basicとadditionalの場所を決定
        # basic, additional共に有音のセグのみを選択対象に
        self.target = self.datafile[np.where((self.datafile[:, b + 2] == 1))[0]]
        while True:
            fail = False
            # anchor
            a_track = random.choice(self.songlist_in_sametempolist)
            a_idx = self.pick_seg_from_track(a_track)
            if a_idx == -1:
                #print(f"\tAnchor does not have any segs. : {a_track}")
                continue
            # positive
            # anchorと同じ曲で前後のセグメントをpositiveとする。overrapを50%しているので2つ前後に
            p_idx = self.pick_p_from_a(a_idx)
            if fail or p_idx == -1:
                #print(f"\tThere is no same song seg to anchor : {self.target[a_idx][0]}")
                #log.append(anchor)
                continue
            # negative
            # 同じテンポでanchorとは違う曲をnegativeとして3曲取り出す
            found = False
            for sametempo in self.sametempolist:
                if int(a_track) in sametempo:
                    #print(type(int(a_track)))
                    # negativeの曲から、2つから1セグメント、もう1つから隣り合う2セグメント取り出す
                    negative_songs = random.sample(sametempo, 3)
                    while int(a_track) in negative_songs: # anchorと曲が被っていたらやり直し
                        negative_songs = random.sample(sametempo, 3)
                    n_track, a2_track, n2_track = negative_songs
                    n_idx  = self.pick_seg_from_track(n_track)
                    a2_idx = self.pick_seg_from_track(a2_track)
                    n2_idx = self.pick_seg_from_track(n2_track)
                    if n_idx == -1 or a2_idx == -1 or n2_idx == -1:
                        fail = True # 無音排除で曲丸ごとない時
                        break
                    p2_idx = self.pick_p_from_a(a2_idx)
                    if p2_idx == -1:
                        fail = True
                    found = True
                    break
            if fail:
                #print(f"\tnegative not found fail : {trackname(n_track)}, {trackname(a2_track)}, {trackname(n2_track)}")
                continue
            if not found:
                print(a_track)
            break
        track_a, track_p, track_n, seg_a, seg_p, seg_n, sound_a, sound_p, sound_n = self.make_pseudo(b, a_idx, p_idx, n_idx, a2_idx, p2_idx, n2_idx)
        return track_a, track_p, track_n, seg_a, seg_p, seg_n, sound_a, sound_p, sound_n, b
        #return track_a, track_n, track_p, seg_a, seg_n, seg_p, sound_a, sound_p, sound_n, a


def main():
    cfg = OmegaConf.create({
        "inst_list": ["drums", "bass", "piano", "guitar", "residuals"]
        })
    triplet_maker = BA4Track_31ways(200, cfg, "train")
    for i in range(20):
        print(triplet_maker())

if __name__ == "__main__":
    main()
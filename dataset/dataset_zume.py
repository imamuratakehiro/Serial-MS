from typing import Any
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pytorch_lightning as pl
#from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as Fa
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

from utils.func import stft, trackname, detrackname, l2normalize, STFT
from model.csn import ConditionalSimNet1d31cases, ConditionalSimNet1d
from .dataset_triplet import BA4Track, BA4Track_31ways, B4TrackInst


def loadseg_from_npz(path):
    npz = np.load(path)
    return npz["wave"].astype(np.float32), npz["sound"]

class LoadSegZume:
    def __init__(self, cfg, mode:str, second, offset, dirpath) -> None:
        self.mode = mode
        self.cfg = cfg
        self.dirpath = dirpath
        self.second = second
        self.offset = offset
    
    def load(self, track_id, seg_id, inst):
        if self.cfg.load_using_librosa:
            stem_path = f"/nas03/assets/Dataset/slakh-2100_2/{trackname(track_id)}/submixes/{inst}.wav"
            stem_wave, sr = librosa.load(
                stem_path,
                sr=self.cfg.sr,
                mono=self.cfg.mono,
                offset=seg_id*self.offset,
                duration=self.second)
        else:
            path = self.dirpath + f"/{inst}/wave{track_id}_{seg_id}.npz"
            stem_wave, sound = loadseg_from_npz(path=path)
        return stem_wave

def max_std(x, axis=None):
    max = abs(x).max(axis=axis, keepdims=True) + 0.000001
    result = x / max
    return result

class CreatePseudo(Dataset):
    """tracklistの曲のseglistのセグメントを読み込んで、擬似楽曲を完成させる。"""
    def __init__(
        self,
        cfg,
        datasettype,
        mode="train",
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        if datasettype == "psd":
            if mode == "train":
                self.second = cfg.seconds_psd_train
                self.offset = cfg.offset_psd_train
            elif mode == "valid":
                self.second = cfg.seconds_psd_valid
                self.offset = cfg.offset_psd_valid
            elif mode == "test":
                self.second = cfg.seconds_psd_test
                self.offset = cfg.offset_psd_test
        elif datasettype == "triplet":
            if mode == "train":
                self.second = cfg.seconds_triplet_train
                self.offset = cfg.offset_triplet_train
            elif mode == "valid":
                self.second = cfg.seconds_triplet_valid
                self.offset = cfg.offset_triplet_valid
            elif mode == "test":
                self.second = cfg.seconds_triplet_test
                self.offset = cfg.offset_triplet_test
        elif datasettype == "c32":
            if mode == "train":
                self.second = cfg.seconds_c32_train
                self.offset = cfg.offset_c32_train
            elif mode == "valid":
                self.second = cfg.seconds_c32_valid
                self.offset = cfg.offset_c32_valid
            elif mode == "test":
                self.second = cfg.seconds_c32_test
                self.offset = cfg.offset_c32_test
        dirpath = f"{self.cfg.dataset_dir}/cutwave/{self.second}s_on{self.offset}"
        self.loader = LoadSegZume(self.cfg, mode, self.second, self.offset, dirpath)

    def load_mix_stems(self, tracklist, seglist, condition=None):
        if condition is None:
            condition = [1 for i in range(len(self.cfg.inst_all))]
        #bin_str = format(condition, f"0{len(self.cfg.inst_list)}b") #2進数化
        stems = []
        if self.cfg.load_using_librosa:
            mix = np.zeros((1, self.second*self.cfg.sr), dtype=np.float32)
        else:
            if self.second == 3:
                mix = np.zeros((1, 131072), dtype=np.float32)
            elif self.second == 10:
                mix = np.zeros((1, 440320), dtype=np.float32)
        #mix = np.zeros((1, 131072), dtype=np.float32)
        for iter, inst in enumerate(self.cfg.inst_all):
            if condition[iter] == 1:
                #Pow = 0
                #path = self.dirpath + f"/{inst}/wave{tracklist[iter]}_{seglist[iter]}.npz"
                #stem_wave, sound = loadseg_from_npz(path=path)
                stem_wave = self.loader.load(tracklist[iter], seglist[iter], inst)
                #print(stem_wave.shape)
                if len(stem_wave) < mix.shape[1] and self.cfg.load_using_librosa:
                    stem_wave = np.pad(stem_wave, (0, mix.shape[1] - len(stem_wave)))
                #Amp = np.abs(stem_wave)
                #Pow = np.mean(Amp**2)
                #if sound == True:
                #    mix_wave = mix_wave / Pow
                if self.cfg.mono:
                    stem_wave = np.expand_dims(stem_wave, axis=0)
                mix = mix[:,:stem_wave.shape[1]] #なぜかshapeが違う。mixは132300(3×44100)なのにstem_waveは131072。なぜ？
                mix += stem_wave; stems.append(stem_wave)
            else:
                if self.cfg.load_using_librosa:
                    stems.append(np.zeros((1, self.second*self.cfg.sr), dtype=np.float32))
                else:
                    if self.second == 3:
                        stems.append(np.zeros((1, 131072), dtype=np.float32))
                    elif self.second == 10:
                        stems.append(np.zeros((1, 440320), dtype=np.float32))
        #mix_wave = max_std(mix_wave) # 混ぜたから正規化してる？分離があるのでなし
        stems = np.stack(stems, axis=0)
        if self.cfg.mix_minus_inst: #inst音源をinst以外の音源に変換
            stems_reverse = []
            for i in range(len(self.cfg.inst_all)):
                stems_reverse.append(np.sum(np.delete(stems, i, 0), axis=0))
            stems = np.stack(stems_reverse, axis=0)
        #mix_param, mix_spec, mix_phase = self.stft.transform(mix)
        # stemはmixと同じparamでnormalize
        #_, stems_spec, stems_phase = self.stft.transform(stems, param=mix_param)
        #if self.mode == "train" or self.mode == "valid":
        #    return mix_spec, stems_spec
        #elif self.mode == "test":
        #    return mix_spec, stems_spec, mix_param, mix_phase
        return mix, stems
    
    def load_mix_stems_librosa(self, tracklist, seglist):
        stems = []
        if self.mode == "train" or self.mode == "valid":
            mix = np.zeros((1, self.cfg.seconds_train*self.cfg.sr), dtype=np.float32)
        elif self.mode == "test":
            mix = np.zeros((1, self.cfg.seconds_test*self.cfg.sr), dtype=np.float32)
        for iter, inst in enumerate(self.cfg.inst_all):
            #Pow = 0
            stem_path = f"/nas03/assets/Dataset/slakh-2100_2/{trackname(tracklist[iter])}/submixes/{inst}.wav"
            stem_wave, sr = librosa.load(stem_path,
                                sr=None,
                                mono=self.cfg.mono,
                                offset=seglist[iter]/self.cfg.sr,
                                duration=self.cfg.seconds_train if self.mode == "train" else self.cfg.seconds_test)
            #Amp = np.abs(stem_wave)
            #Pow = np.mean(Amp**2)
            #if sound == True:
            #    mix_wave = mix_wave / Pow
            if self.cfg.mono:
                stem_wave = np.expand_dims(stem_wave, axis=0)
            mix = mix[:,:stem_wave.shape[1]] #なぜかshapeが違う。mixは132300(3×44100)なのにstem_waveは131072。なぜ？
            mix += stem_wave; stems.append(stem_wave)
        #mix_wave = max_std(mix_wave) # 混ぜたから正規化してる？分離があるのでなし
        stems = np.stack(stems, axis=0)
        mix_param, mix_spec, mix_phase = stft(mix, self.cfg)
        # stemはmixと同じparamでnormalize
        _, stems_spec, stems_phase = stft(stems, self.cfg, param=mix_param)
        return mix_spec, stems_spec

class SongDataForPreTrain(Dataset):
    """事前学習用のdataset"""
    def __init__(
        self,
        cfg,
        mode="train"
        ) -> None:
        super().__init__()
        self.datafile = pd.read_csv(cfg.metadata_dir + "zume/slakh/single3_200data-euc_zero.csv", index_col=0).values
        self.loader = CreatePseudo(cfg, datasettype="triplet", mode=mode)
        self.cfg = cfg
    
    def load_embvec(self, track_id, seg_id, condition=None):
        #bin_str = format(condition, f"0{len(self.cfg.inst_list)}b") #2進数化
        if condition is None:
            condition = [1 for _ in range(len(self.cfg.inst_all))]
        dirpath = f"/nas03/assets/Dataset/slakh/single3_200data-euc_zero/"
        embvec = []
        for idx, inst in enumerate(self.cfg.inst_all):
            if condition[idx] == 1:
                vec_inst = np.load(dirpath + inst + f"/Track{track_id}/seg{seg_id}.npy")
                #print(inst, np.max(vec_inst))
            else:
                vec_inst = np.zeros((1,128), dtype=np.float32)
            if self.cfg.normalize128:
                vec_inst = l2normalize(vec_inst)
            embvec.append(vec_inst)
        return torch.from_numpy(l2normalize(np.concatenate(embvec, axis=1).squeeze()))

    def load_embvec_640(self, track_id, seg_id, condition=0b11111):
        """全ての音源のembを読んで640次元で正規化してからcsnでconditionの部分だけ残してreturn。"""
        #bin_str = format(condition, f"0{len(self.cfg.inst_list)}b") #2進数化
        dirpath = f"/nas03/assets/Dataset/slakh/single3_200data-euc_zero/"
        embvec = self.load_embvec(track_id, seg_id)
        csn = ConditionalSimNet1d31cases(self.cfg)
        return csn(embvec, torch.tensor([condition], device=embvec.device))
    
    def load_embvec_stems(self, track_id, seg_id, condition=None):
        if self.cfg.emb_640norm:
            embvec = self.load_embvec(track_id, seg_id)
            csn = ConditionalSimNet1d()
            embvec_inst = []
            for idx,inst in enumerate(self.cfg.inst_all):
                embvec_inst.append(csn(embvec, torch.tensor([idx], device=embvec.device)))
            return torch.stack(embvec_inst, dim=0).squeeze()
            #return torch.stack([csn(embvec, torch.tensor([i], device=embvec.device)) for i in range(len(self.cfg.inst_list))], dim=0).squeeze()
        else:
            if condition is None:
                condition = [1 for i in range(len(self.cfg.inst_all))]
            #bin_str = format(condition, f"0{len(self.cfg.inst_list)}b") #2進数化
            dirpath = f"/nas03/assets/Dataset/slakh/single3_200data-euc_zero/"
            embvec = []
            for idx, inst in enumerate(self.cfg.inst_all):
                vec_inst = np.zeros((1,640), dtype=np.float32)
                if condition[idx] == 1:
                    vec_inst[:,idx*128:(idx+1)*128] = np.load(dirpath + inst + f"/Track{track_id}/seg{seg_id}.npy")
                else:
                    pass
                    #vec_inst = np.zeros((1,128), dtype=np.float32)
                #if self.cfg.normalize128:
                #    vec_inst = l2normalize(vec_inst)
                embvec.append(l2normalize(vec_inst.squeeze()))
            return torch.from_numpy(np.stack(embvec, axis=0))
    
    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, index):
        track_id, seg_id = self.datafile[index, 0], self.datafile[index, 1]
        tracklist = [track_id for _ in self.cfg.inst_all]
        seglist   = [seg_id   for _ in self.cfg.inst_all]
        if self.cfg.condition32:
            condition_b = random.randrange(0, 2**len(self.cfg.inst_all))
            condition = [int(i) for i in format(condition_b, f"0{len(self.cfg.inst_all)}b")]
            mix_spec, _ = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist, condition=condition)
            if self.cfg.emb_640norm:
                return mix_spec, condition, self.load_embvec_640(track_id, seg_id, condition=condition_b).squeeze()
            else:
                return mix_spec, condition, self.load_embvec(track_id, seg_id, condition=condition)
        else:
            mix_spec, stems_spec = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist)
            return mix_spec, stems_spec, self.load_embvec(track_id, seg_id), self.load_embvec_stems(track_id, seg_id)

def load_lst(listpath):
    data = []
    #listdir = cfg.metadata_dir + "lst"
    #with open(f"{listdir}/{listname}", "r") as f:
    with open(listpath, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            groups = line.split(";")
            items = []
            for group in groups:
                numbers = [int(num.strip()) for num in group.split(",")]
                items.append(numbers)
            data.append(tuple(items))
    return data

class PsdLoader(Dataset):
    """test用擬似楽曲をロード"""
    def __init__(
        self,
        cfg,
        inst,
        mode:str,
        psd_mine: bool
    ):
        lst_dir = cfg.metadata_dir + f"lst/"
        self.cfg = cfg
        #if cfg.test_psd_mine:
        if psd_mine:
            self.data = load_lst(lst_dir + f"psd_{mode}_{inst}_10_10.lst")
        else:
            if mode == "valid":
                self.data = load_lst(lst_dir + f"psds_{mode}_10_{inst}.lst")
            elif mode == "test":
                self.data = load_lst(lst_dir + f"psds_{mode}_{inst}.lst")
        self.loader = CreatePseudo(cfg, datasettype="psd", mode=mode)
        self.condition = cfg.inst_all.index(inst)
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        c_selected, tracklist, seglist, ID_ver = self.data[index]
        ID = ID_ver[0]; ver = ID_ver[1]
        #ver = f"{ID}-{ver}"
        #if self.mode == "test":
        #    data, _, _, _ = self.loader.load_mix_stems(tracklist, seglist)
        #else:
        #    data, _ = self.loader.load_mix_stems(tracklist, seglist)
        data_psd, _ = self.loader.load_mix_stems(tracklist, seglist)
        """data_not_psd, _ = self.loader.load_mix_stems(
            tracklist=[tracklist[c_selected[0]] for i in range(len(self.cfg.inst_all))],
            seglist=[seglist[c_selected[0]] for i in range(len(self.cfg.inst_all))])"""
        return ID, ver, seglist[c_selected[0]], data_psd, self.condition

class TestLoader(Dataset):
    """test用擬似楽曲でない曲をロード"""
    def __init__(
        self,
        cfg,
        inst,
        mode:str,
    ):
        lst_dir = cfg.metadata_dir + f"zume/slakh/"
        self.cfg = cfg
        self.data = self.make_test_data(inst, mode)
        self.loader = CreatePseudo(cfg, datasettype="psd", mode=mode)
        self.condition = cfg.inst_all.index(inst)
        self.mode = mode
    
    def make_test_data(self, inst, mode):
        lst_dir = self.cfg.metadata_dir + f"zume/slakh/"
        if mode == "test":
            metadata = pd.read_csv(self.cfg.metadata_dir + f"zume/slakh/10s_on10.0_with_silence.csv").values
            songdata = json.load(open(self.cfg.metadata_dir + f"slakh/test_redux_136.json", 'r'))
        elif mode == "valid":
            metadata = pd.read_csv(self.cfg.metadata_dir + f"zume/slakh/10s_on5.0_with_silence.csv").values
            songdata = json.load(open(self.cfg.metadata_dir + f"slakh/valid_redux.json", 'r'))
        data = []
        picked_id = random.sample(songdata, self.cfg.n_song_test)
        for id in picked_id:
            #print(id, type(id))
            track_all = metadata[np.where((metadata[:, 0] == id) & (metadata[:, self.cfg.inst_all.index(inst) + 2] == 1))[0]]
            for track in track_all:
                data.append(
                    [
                        [self.cfg.inst_all.index(inst)], # c_selected
                        [track[0] for _ in self.cfg.inst_all], # tracklist
                        [track[1] for _ in self.cfg.inst_all], # seglist
                        [track[0], track[0]] # ID_ver
                    ]
                )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        c_selected, tracklist, seglist, ID_ver = self.data[index]
        ID = ID_ver[0]; ver = ID_ver[1]
        #ver = f"{ID}-{ver}"
        #if self.mode == "test":
        #    data, _, _, _ = self.loader.load_mix_stems(tracklist, seglist)
        #else:
        #    data, _ = self.loader.load_mix_stems(tracklist, seglist)
        data_psd, _ = self.loader.load_mix_stems(tracklist, seglist)
        return ID, ver, seglist[c_selected[0]], data_psd, self.condition

class TripletLoaderFromList(Dataset):
    def __init__(
        self,
        cfg,
        mode,
        inst=None,
        ):
        self.cfg = cfg
        lst_dir = cfg.metadata_dir + f"lst/"
        if mode == "train":
            #self.triplets = load_lst(f"triplets_1200_ba1_4_withsil_20000triplets.lst")
            if self.cfg.pseudo == "ba_4t":
                self.triplets = load_lst(lst_dir + f"triplets_1200_ba4t_withsil_nosegsfl_20000triplets.lst")
            elif self.cfg.pseudo == "31ways":
                self.triplets = load_lst(lst_dir + f"triplets_ba4t_31ways_train_3s_10000songs.lst")
            elif self.cfg.pseudo == "b_4t_inst":
                self.triplets = load_lst(lst_dir + f"triplets_b4t_{inst}_train_3s_10000songs.lst")
        elif mode == "valid":
            #self.triplets = load_lst(f"triplets_valid_ba1_4_withsil_200triplets.lst")
            if self.cfg.pseudo == "ba_4t":
                self.triplets = load_lst(lst_dir + f"triplets_valid_ba4t_withsil_nosegsfl_200triplets.lst")
            elif self.cfg.pseudo == "31ways":
                self.triplets = load_lst(lst_dir + f"triplets_ba4t_31ways_valid_10s_2000songs.lst")
            elif self.cfg.pseudo == "b_4t_inst":
                self.triplets = load_lst(lst_dir + f"triplets_b4t_{inst}_valid_10s_200songs.lst")
        self.loader = CreatePseudo(cfg, datasettype="triplet", mode=mode)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        # print(self.triplets[index])
        (
            tracklist_a,
            tracklist_p,
            tracklist_n,
            seglist_a,
            seglist_p,
            seglist_n,
            sound_a,
            sound_p,
            sound_n,
            c,
        ) = self.triplets[index]
        mix_a, stems_a = self.loader.load_mix_stems(tracklist_a, seglist_a, sound_a)
        mix_p, stems_p = self.loader.load_mix_stems(tracklist_p, seglist_p, sound_p)
        mix_n, stems_n = self.loader.load_mix_stems(tracklist_n, seglist_n, sound_n)
        # cのindexにおいてanchor、positive、negativeの曲のセグメント
        return mix_a, stems_a, mix_p, stems_p, mix_n, stems_n, torch.FloatTensor(sound_a), torch.FloatTensor(sound_p), torch.FloatTensor(sound_n), c[0]

class TripletLoaderEachTime(Dataset):
    def __init__(
        self,
        cfg,
        mode,
        inst=None,
        ):
        self.cfg = cfg
        if mode == "train":
            self.n_triplet = self.cfg.n_triplet_train
        elif mode == "valid":
            self.n_triplet = self.cfg.n_triplet_valid
        if self.cfg.pseudo == "ba_4t":
            self.triplet_maker = BA4Track(n_triplet=self.n_triplet, cfg=cfg, mode=mode)
        elif self.cfg.pseudo == "31ways":
            self.triplet_maker = BA4Track_31ways(n_triplet=self.n_triplet, cfg=cfg, mode=mode)
        elif self.cfg.pseudo == "b_4t_inst":
            self.triplet_maker = B4TrackInst(n_triplet=self.n_triplet, cfg=cfg, mode=mode)
        else:
            raise NotImplementedError
        self.loader = CreatePseudo(cfg, datasettype="triplet", mode=mode)

    def __len__(self):
        return self.n_triplet

    def __getitem__(self, index):
        # print(self.triplets[index])
        (
            tracklist_a,
            tracklist_p,
            tracklist_n,
            seglist_a,
            seglist_p,
            seglist_n,
            sound_a,
            sound_p,
            sound_n,
            c, # [b, a]の順で入っている
        ) = self.triplet_maker()
        mix_a, stems_a = self.loader.load_mix_stems(tracklist_a, seglist_a, sound_a)
        mix_p, stems_p = self.loader.load_mix_stems(tracklist_p, seglist_p, sound_p)
        mix_n, stems_n = self.loader.load_mix_stems(tracklist_n, seglist_n, sound_n)
        # cのindexにおいてanchor、positive、negativeの曲のセグメント
        return mix_a, stems_a, mix_p, stems_p, mix_n, stems_n, torch.FloatTensor(sound_a), torch.FloatTensor(sound_p), torch.FloatTensor(sound_n), torch.tensor(c)

class Condition32Loader(Dataset):
    def __init__(
        self,
        cfg,
        mode,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        if mode == "train":
            self.datafile = pd.read_csv(cfg.metadata_dir + "zume/slakh/3s_on1.5_no_silence_all.csv", index_col=0).values[:104663]
        elif mode == "valid":
            self.datafile = pd.read_csv(cfg.metadata_dir + "zume/slakh/10s_on5.0_no_silence_all.csv", index_col=0).values[44683:]
        elif mode == "test":
            self.datafile = pd.read_csv(cfg.metadata_dir + "zume/slakh/10s_on10.0_no_silence_all.csv", index_col=0).values
        self.loader = CreatePseudo(cfg, datasettype="c32", mode=mode)
        self.mode = mode
    
    def __len__(self):
        if self.mode == "train" or self.mode == "valid":
            return len(self.datafile)
        elif self.mode == "test":
            return self.cfg.n_dataset_test
    
    def __getitem__(self, index):
        track_id, seg_id = self.datafile[index, 0], self.datafile[index, 1]
        tracklist = [track_id for _ in self.cfg.inst_all]
        seglist   = [seg_id   for _ in self.cfg.inst_all]
        condition = random.randrange(0, 2**len(self.cfg.inst_all))
        cases = format(condition, f"0{len(self.cfg.inst_all)}b") #2進数化
        condition = [int(i) for i in cases]
        cases = torch.tensor([float(int(i)) for i in cases])
        #if self.mode == "train" or self.mode == "valid":
        #    mix_spec, stems_spec = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist, condition=condition)
        #    return mix_spec, stems_spec, cases
        #elif self.mode == "test":
        #    mix_spec, stems_spec, param, phase = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist, condition=condition)
        #    return mix_spec, stems_spec, param, phase, cases
        mix, stems = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist, condition=condition)
        return mix, stems, cases

class SongDataForUNetNotPseudo(Dataset):
    """mix音源と指定した楽器音のstems音源をそれぞれreturnする"""
    def __init__(
        self,
        cfg,
        mode) -> None:
        super().__init__()
        self.path = "/nas03/assets/Dataset/slakh-2100_2"
        if mode == "train":
            self.datafile = np.loadtxt(cfg.metadata_dir + f"slakh/{cfg.seconds_unet_train}s_no_silence_or0.5_0.25/{mode}_slakh_{cfg.seconds_unet_train}s_stems.txt", delimiter = ",", dtype = "unicode")
            self.second = cfg.seconds_unet_train
        elif mode == "valid":
            self.datafile = np.loadtxt(cfg.metadata_dir + f"slakh/{cfg.seconds_unet_valid}s_no_silence_or0.5_0.25/{mode}_slakh_{cfg.seconds_unet_valid}s_stems.txt", delimiter = ",", dtype = "unicode")
            idx = np.random.choice(self.datafile.shape[0], cfg.n_dataset_valid, replace=False)
            self.datafile = self.datafile[idx]
            self.second = cfg.seconds_unet_valid
        elif mode == "test":
            self.datafile = np.loadtxt(cfg.metadata_dir + f"slakh/{cfg.seconds_unet_test}s_no_silence_or0.5_0.25/{mode}_slakh_{cfg.seconds_unet_test}s_stems.txt", delimiter = ",", dtype = "unicode")
            idx = np.random.choice(self.datafile.shape[0], cfg.n_dataset_test, replace=False)
            self.datafile = self.datafile[idx]
            self.second = cfg.seconds_unet_test
        self.cfg = cfg
        self.mode = mode

    def __len__(self):
        return self.datafile.shape[0]

    def __getitem__(self, idx):
        n = int(self.datafile[idx][1])
        # mix音源
        mix_path = self.path + "/" + self.datafile[idx][0] + "/mix.flac"
        mix_wave, _ = librosa.load(
            mix_path,
            sr=self.cfg.sr,
            mono=self.cfg.mono,
            offset=n/self.cfg.sr,
            duration=self.second)
        if self.cfg.mono:
            mix_wave = np.expand_dims(mix_wave, axis=0)

        # stem音源
        stems_wave = []
        for i, inst in enumerate(self.cfg.inst_list):
            stem_path = self.path + "/" + self.datafile[idx][0] + f"/submixes/{inst}.wav"
            stem_wave, _ = librosa.load(
                stem_path,
                sr=self.cfg.sr,
                mono=self.cfg.mono,
                offset=n/self.cfg.sr,
                duration=self.second)
            if self.cfg.mono:
                stem_wave = np.expand_dims(stem_wave, axis=0)
            stems_wave.append(stem_wave)
        stems_wave = np.stack(stems_wave, axis=0)
        return mix_wave, stems_wave

class SongDataForUNetPseudo(Dataset):
    """UNetの音源分離学習用データとして擬似楽曲を使用"""
    def __init__(
        self,
        cfg,
        mode,
        inst=None,
        ):
        self.cfg = cfg
        if mode == "train":
            self.n_triplet = self.cfg.n_triplet_train
        elif mode == "valid":
            self.n_triplet = self.cfg.n_triplet_valid
        self.triplet_maker = B4TrackInst(n_triplet=self.n_triplet, cfg=cfg, mode=mode)
        self.loader = CreatePseudo(cfg, datasettype="triplet", mode=mode)

    def __len__(self):
        return self.n_triplet

    def __getitem__(self, index):
        # print(self.triplets[index])
        (
            tracklist_a,
            tracklist_p,
            tracklist_n,
            seglist_a,
            seglist_p,
            seglist_n,
            sound_a,
            sound_p,
            sound_n,
            c, # [b, a]の順で入っている
        ) = self.triplet_maker()
        mix_a, stems_a = self.loader.load_mix_stems(tracklist_a, seglist_a, sound_a)
        #mix_p, stems_p = self.loader.load_mix_stems(tracklist_p, seglist_p, sound_p)
        #mix_n, stems_n = self.loader.load_mix_stems(tracklist_n, seglist_n, sound_n)
        # anchor部分のみ使用
        return mix_a, stems_a[self.cfg.inst_all.index(self.cfg.inst)]
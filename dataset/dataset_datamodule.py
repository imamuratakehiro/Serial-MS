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
from .dataset_triplet import TripletDatasetOneInst, LoadSongWithLabel
from .dataset_zume import CreatePseudo, SongDataForPreTrain, PsdLoader, TripletLoader, Condition32Loader


class PreTrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.mylogger = MyLoggerModel()
    
    def prepare_data(self) -> None:
        print(f"\n----------------------------------------")
        print(f"Use dataset {self.cfg.datasetname}.")
        print(f"The frame size is setted to {self.cfg.f_size}.")
        print("----------------------------------------")
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.mylogger.s_dataload()
            self.trainset = SongDataForPreTrain(mode="train", cfg=self.cfg)
            self.validset = [PsdLoader(mode="valid", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            self.mylogger.f_dataload()
        if stage == "test" or stage is None:
            self.testset = [PsdLoader(mode="test", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
        if stage == "predict"  or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.batch_train,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.validset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.cfg.inst_list))]
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.testset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.cfg.inst_list))]
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class TripletDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.mylogger = MyLoggerModel()
    
    def prepare_data(self) -> None:
        print(f"\n----------------------------------------")
        print(f"Use dataset {self.cfg.datasetname}.")
        print(f"The frame size is setted to {self.cfg.f_size}.")
        print("----------------------------------------")
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.mylogger.s_dataload()
            self.trainset = TripletLoader(mode="train", cfg=self.cfg)
            self.validset = [PsdLoader(mode="valid", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            self.validset.insert(0, TripletLoader(mode="valid", cfg=self.cfg))
            self.mylogger.f_dataload()
        if stage == "test" or stage is None:
            self.testset = [PsdLoader(mode="test", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            self.testset.insert(0, Condition32Loader(mode="test", cfg=self.cfg))
        if stage == "predict"  or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.batch_train,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.validset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.validset))]
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.testset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.testset))]
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

class TripletDataModuleBSRnn(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.mylogger = MyLoggerModel()
    
    def prepare_data(self) -> None:
        print(f"\n----------------------------------------")
        print(f"Use dataset {self.cfg.datasetname}.")
        print(f"The frame size is setted to {self.cfg.f_size}.")
        print("----------------------------------------")
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.mylogger.s_dataload()
            self.trainset = TripletLoader(mode="train", cfg=self.cfg, inst=self.cfg.inst)
            self.validset = [PsdLoader(mode="valid", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            self.validset.insert(0, TripletLoader(mode="valid", cfg=self.cfg, inst=self.cfg.inst))
            self.mylogger.f_dataload()
        if stage == "test" or stage is None:
            self.testset = [PsdLoader(mode="test", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            self.testset.insert(0, Condition32Loader(mode="test", cfg=self.cfg))
        if stage == "predict"  or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.batch_train,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.validset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.validset))]
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.testset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.testset))]
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


class TripletDataModuleOneInst(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.mylogger = MyLoggerModel()
    
    def prepare_data(self) -> None:
        print(f"\n----------------------------------------")
        print(f"Use dataset {self.cfg.datasetname}.")
        print(f"The seg is reduced silence with {self.cfg.reduce_silence}.")
        print(f"The frame size is setted to {self.cfg.f_size}.")
        print(f"The length of seg for train is {self.cfg.seconds_train}s.")
        print(f"The length of seg for valid and test is {self.cfg.seconds_test}s.")
        print("----------------------------------------")
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.mylogger.s_dataload()
            self.trainset = TripletDatasetOneInst(mode="train", cfg=self.cfg)
            if self.cfg.valid_knn:
                self.validset = LoadSongWithLabel(mode="valid", cfg=self.cfg)
            else:
                self.validset = TripletDatasetOneInst(mode="valid", cfg=self.cfg)
            self.mylogger.f_dataload()
        if stage == "test" or stage is None:
            self.testset = LoadSongWithLabel(mode="test", cfg=self.cfg)
        if stage == "predict"  or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.batch,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.validset,
            batch_size=self.cfg.batch,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.testset,
            batch_size=self.cfg.batch,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
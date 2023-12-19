from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ModuleList as MList, ModuleDict as MDict
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm

from utils.func import file_exist, knn_psd, tsne_psd, istft, tsne_psd_marker, TorchSTFT, tsne_not_psd
from ..csn import ConditionalSimNet1d
from ..tripletnet import CS_Tripletnet


class NNet(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        featurenet: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cfg,
        ckpt_model_path_unet,
        ckpt_model_path_featurenet
        ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # network
        self.unet = unet
        self.featurenet = featurenet
        # loading pretrained model
        model_checkpoint_unet = {}
        if ckpt_model_path_unet is not None:
            print("== Loading pretrained model (unet)...")
            checkpoint = torch.load(ckpt_model_path_unet)
            """for key in checkpoint["state_dict"]:
                model_checkpoint_unet[key.replace("net.","")] = checkpoint["state_dict"][key]
            self.unet.load_state_dict(model_checkpoint_unet)"""
            for key in checkpoint["state_dict"]:
                if "unet." in key:
                    model_checkpoint_unet[key.replace("unet.","")] = checkpoint["state_dict"][key]
            self.unet.load_state_dict(model_checkpoint_unet)
            print("== pretrained model was loaded!")
        model_checkpoint_featurenet = {}
        if ckpt_model_path_featurenet is not None:
            print("== Loading pretrained model (featurenet)...")
            checkpoint = torch.load(ckpt_model_path_featurenet)
            for key in checkpoint["state_dict"]:
                model_checkpoint_featurenet[key.replace("net.","")] = checkpoint["state_dict"][key]
            self.featurenet.load_state_dict(model_checkpoint_featurenet)
            print("== pretrained model was loaded!")
        print(self.featurenet)
        # model's required grad
        if not cfg.unet_required_grad:
            for param in unet.parameters():
                param.requires_grad = False
        if not cfg.featurenet_required_grad:
            for param in featurenet.parameters():
                param.requires_grad = False
        print(self.unet)
        print(self.featurenet)

        # loss function
        self.loss_unet  = nn.L1Loss(reduction="mean")
        self.loss_triplet = nn.MarginRankingLoss(margin=cfg.margin, reduction="none") #バッチ平均
        self.loss_l2      = nn.MSELoss(reduction="sum")
        self.loss_mrl     = nn.MarginRankingLoss(margin=cfg.margin, reduction="mean") #バッチ平均
        self.loss_cross_entropy = nn.BCEWithLogitsLoss(reduction="mean")

        # metric objects for calculating and averaging accuracy across batches
        #self.train_acc = Accuracy(task="multiclass", num_classes=10)
        #self.val_acc = Accuracy(task="multiclass", num_classes=10)
        #self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        """
        self.song_type = ["anchor", "positive", "negative", "cases", "all"]
        # train
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.train_loss_unet = {type: MeanMetric() for type in self.song_type}
        self.train_loss_triplet = {inst: MeanMetric() for inst in cfg.inst_list}
        self.train_loss_recog = MeanMetric()
        self.train_recog_acc = {inst: Accuracy for inst in cfg.inst_list}
        self.train_dist_p = {inst: MeanMetric() for inst in cfg.inst_list}
        self.train_dist_n = {inst: MeanMetric() for inst in cfg.inst_list}
        # validate
        self.valid_loss_unet = {type: MeanMetric() for type in self.song_type}
        self.valid_loss_triplet = {inst: MeanMetric() for inst in cfg.inst_list}
        self.valid_loss_recog = MeanMetric()
        self.valid_recog_acc = {inst: Accuracy for inst in cfg.inst_list}
        self.valid_dist_p = {inst: MeanMetric() for inst in cfg.inst_list}
        self.valid_dist_n = {inst: MeanMetric() for inst in cfg.inst_list}
        """
        self.song_type = ["anchor", "positive", "negative"]
        self.recorder = MDict({})
        for step in ["Train", "Valid"]:
            self.recorder[step] = MDict({})
            self.recorder[step]["loss_all"] = MeanMetric()
            self.recorder[step]["loss_unet"] = MDict({type: MeanMetric() for type in self.song_type})
            self.recorder[step]["loss_unet"]["all"] = MeanMetric()
            self.recorder[step]["loss_triplet"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
            self.recorder[step]["loss_triplet"]["all"] = MeanMetric()
            self.recorder[step]["loss_recog"] = MeanMetric()
            self.recorder[step]["recog_acc"] = MDict({inst: BinaryAccuracy() for inst in cfg.inst_list})
            self.recorder[step]["dist_p"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
            self.recorder[step]["dist_n"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
        self.recorder["Test"] = MDict({})
        self.recorder["Test"]["recog_acc"] = MDict({inst: BinaryAccuracy() for inst in cfg.inst_list})
        self.recorder_psd = {}
        self.n_sound = {}
        for step in ["Valid", "Test"]:
            self.recorder_psd[step] = {}
            self.recorder_psd[step]["label"] = {}
            self.recorder_psd[step]["vec"] = {}
            self.n_sound[step] = {}
            for psd in ["psd", "not_psd", "psd_mine"]:
                self.recorder_psd[step]["label"][psd] = {inst: [] for inst in cfg.inst_list}
                self.recorder_psd[step]["vec"][psd] = {inst: [] for inst in cfg.inst_list}
                self.n_sound[step][psd] = {inst: 0 for inst in cfg.inst_list}
        #self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        #self.val_acc_best = MaxMetric()
        self.stft = TorchSTFT(cfg=cfg)
        self.cfg = cfg
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        #self.val_loss.reset()
        #self.val_acc.reset()
        #self.val_acc_best.reset()
        self.recorder["Valid"]["loss_all"].reset()
        for type in self.song_type:
            self.recorder["Valid"]["loss_unet"][type].reset()
        self.recorder["Valid"]["loss_unet"]["all"].reset()
        for inst in self.cfg.inst_list:
            self.recorder["Valid"]["loss_triplet"][inst].reset()
            self.recorder["Valid"]["loss_recog"][inst].reset()
            self.recorder["Valid"]["recog_acc"][inst].reset()
            self.recorder["Valid"]["dist_p"][inst].reset()
            self.recorder["Valid"]["dist_n"][inst].reset()
        self.recorder["Valid"]["loss_triplet"]["all"].reset()
    """
    def dataload_triplet(self, batch):
        # データセットを学習部分と教師部分に分けてdeviceを設定
        #self.logger.s_dataload()
        #print(f"\t..................dataset log.......................")
        anchor_X, anchor_y, positive_X, positive_y, negative_X, negative_y, triposi, posiposi = batch
        anchor_y   = torch.permute(anchor_y,   (1, 0, 2, 3, 4))
        positive_y = torch.permute(positive_y, (1, 0, 2, 3, 4))
        negative_y = torch.permute(negative_y, (1, 0, 2, 3, 4))
        #print(f"\t....................................................")
        #self.logger.f_dataload()
        return anchor_X, anchor_y, positive_X, positive_y, negative_X, negative_y, triposi, posiposi
    
    def dataload_32cases(self, cases32_loader):
        # 32situationをロード
        cases_X, cases_y, cases = cases32_loader.load()
        cases_y = torch.permute(cases_y, (1, 0, 2, 3, 4))
        return cases_X, cases_y, cases
    """
    
    def get_loss_unet(self, X, y, pred_mask):
        batch = X.shape[0]
        loss = 0
        for idx, inst in enumerate(self.cfg.inst_list): #個別音源でロスを計算
            pred = X * pred_mask[inst]
            loss += self.loss_unet(pred, y[:,idx])
        return loss / len(self.cfg.inst_list)
    
    def get_loss_unet_triposi(self, y, pred, triposi):
        # triplet positionのところのみ分離ロスを計算
        batch = y.shape[0]
        loss = 0
        for idx, c in enumerate(triposi): #個別音源でロスを計算
            #if len(self.cfg.inst_list) == 1:
            #    loss += self.loss_unet(pred[self.cfg.inst_all[c.item()]][idx], y[idx, 0])
            #else:
            loss += self.loss_unet(pred[self.cfg.inst_all[c.item()]][idx], y[idx, c])
        return loss / batch
    
    """
    def get_loss_triplet(self, e_a, e_p, e_n, triposi):
        batch = triposi.shape[0]
        loss_all = 0
        loss = {inst: 0 for inst in self.cfg.inst_list}
        dist_p_all = {inst: 0 for inst in self.cfg.inst_list}
        dist_n_all = {inst: 0 for inst in self.cfg.inst_list}
        csn = ConditionalSimNet1d(batch=1)
        inst_n_triplet = [0 for i in range(len(self.cfg.inst_list))]
        for b, i in enumerate(triposi):
            condition = self.cfg.inst_list[i.item()]
            masked_e_a = csn(e_a[b], torch.tensor([i], device=e_a.device))
            masked_e_p = csn(e_p[b], torch.tensor([i], device=e_p.device))
            masked_e_n = csn(e_n[b], torch.tensor([i], device=e_n.device))
            dist_p = F.pairwise_distance(masked_e_a, masked_e_p, 2)
            dist_n = F.pairwise_distance(masked_e_a, masked_e_n, 2)
            target = torch.ones_like(dist_p).to(dist_p.device) #1で埋める
            # トリプレットロス
            triplet = self.loss_triplet(dist_n, dist_p, target) #3つのshapeを同じにする
            loss[condition] += triplet.item()
            loss_all += triplet
            dist_p_all[condition] += dist_p.item()
            dist_n_all[condition] += dist_n.item()
            inst_n_triplet[i] += 1
        # lossを出現回数で割って平均の値に
        for i in range(len(self.cfg.inst_list)):
            condition = self.cfg.inst_list[i]
            if inst_n_triplet[i] != 0:
                loss[condition] /= inst_n_triplet[i]
                dist_p_all[condition] /= inst_n_triplet[i]
                dist_n_all[condition] /= inst_n_triplet[i]
        return loss_all/batch, loss, dist_p_all, dist_n_all
    """
    def get_loss_triplet(self, e_a, e_p, e_n, triposi):
        #batch = triposi.shape[0]
        if len(self.cfg.inst_list) == 1:
            distp = F.pairwise_distance(e_a, e_p, 2)
            distn = F.pairwise_distance(e_a, e_n, 2)
        else:
            tnet = CS_Tripletnet(ConditionalSimNet1d().to(e_a.device))
            distp, distn = tnet(e_a, e_p, e_n, triposi)
        target = torch.FloatTensor(distp.size()).fill_(1).to(distp.device) # 1で埋める
        loss = self.loss_triplet(distn, distp, target) # トリプレットロス
        loss_all = torch.sum(loss)/len(triposi)
        loss_inst  = {inst: torch.sum(loss[torch.where(triposi==i)])/len(torch.where(triposi==i)[0])  if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_all)}
        dist_p_all = {inst: torch.sum(distp[torch.where(triposi==i)])/len(torch.where(triposi==i)[0]) if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_all)}
        dist_n_all = {inst: torch.sum(distn[torch.where(triposi==i)])/len(torch.where(triposi==i)[0]) if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_all)}
        return loss_all, loss_inst, dist_p_all, dist_n_all
    
    def get_loss_recognise1(self, emb, cases, l = 1e5):
        batch = len(cases)
        loss_emb = 0
        #zero = torch.tensor(0).to(device)
        #one  = torch.tensor(1).to(device)
        for b in range(batch):
            for c, inst in enumerate(self.inst_list):
                if cases[b][c] == "1":
                    # 0ベクトルとembedded vectorの距離
                    dist_0 = F.pairwise_distance(emb[inst][b], torch.zeros_like(emb[inst][b], device=cases.device), 2)
                    target = torch.ones_like(dist_0).to(cases.device) #1で埋める
                    # 0ベクトルとembedded vectorの距離がmarginよりも大きくなることを期待
                    loss_emb += self.loss_mrl(dist_0, torch.zeros_like(dist_0, device=cases.device), target)
                    #loss_emb_zero += self.loss_fn(torch.mean(torch.abs(emb[inst][b])), one)
                else:
                    loss_emb += self.loss_l2(emb[inst][b], torch.zeros_like(emb[inst][b], device=cases.device))
        #batch = X.shape[0]
        return loss_emb / batch # バッチ平均をとる

    def get_loss_recognise2(self, probability, cases):
        batch = len(cases)
        loss_recognise = 0
        for b in range(batch):
            for c, inst in enumerate(self.cfg.inst_list):
                # 実際の有音無音判定と予想した有音確率でクロスエントロピーロスを計算
                loss_recognise += self.loss_cross_entropy(probability[inst][b], cases[b, c])
        return loss_recognise / batch / len(self.cfg.inst_list)
    
    def stft_complex(self, wave):
        x = self.stft.stft(wave)
        *other, C, F, T = x.shape
        return torch.view_as_real(x.reshape(-1, C, F, T)).permute(0, 1, 4, 2, 3).reshape(*other, C * 2, F, T)
    
    def transform_for_unet(self, wave):
        device = wave.device
        if self.cfg.complex_unet:
            return {"spec": self.stft_complex(wave)}
        else:
            x = self.stft.stft(wave)
            x, phase = self.stft.magphase(x)
        return {"spec": x, "phase": phase}
    
    def transform_for_featurenet(self, **kwargs):
        if self.cfg.complex_unet and self.cfg.complex_featurenet:
            """complex -> complex"""
            return torch.concat(list(kwargs["unet_out"].values()), dim=1)
        elif self.cfg.complex_unet and not self.cfg.complex_featurenet:
            """complex -> spec"""
            x, _ = self.stft.magphase(torch.concat(list(kwargs["unet_out"].values()), dim=1))
            if self.cfg.mel_featurenet:
                x = self.stft.mel(x)
            if self.cfg.db_featurenet:
                x = self.stft.amp2db(x)
            return x
        elif not self.cfg.complex_unet and self.cfg.complex_featurenet:
            """spec -> comlex"""
            out = []
            for inst in self.cfg.inst_list:
                out.append(kwargs["unet_out"][inst] * kwargs["phase"])
            return torch.stack(out, dim=1)
        elif not self.cfg.complex_unet and not self.cfg.complex_featurenet:
            """spec -> spec"""
            x = torch.concat(list(kwargs["unet_out"].values()), dim=1)
            if self.cfg.mel_featurenet:
                x = self.stft.mel(x)
            if self.cfg.db_featurenet:
                x = self.stft.amp2db(x)
            return x
    """
    def clone_for_additional(self, a_x, a_y, p_x, p_y, n_x, n_y, s_a, s_p, s_n, triposi):
        if triposi.dim() == 2: # [b, a]で入ってる
            x_a, x_p, x_n, y_a, y_p, y_n, a_s, p_s, n_s, tp = [], [], [], [], [], [], [], [], [], []
            for i, ba in enumerate(triposi):
                # basic
                x_a.append(a_x[i].clone()); x_p.append(p_x[i].clone()); x_n.append(n_x[i].clone())
                y_a.append(a_y[i].clone()); y_p.append(p_y[i].clone()); y_n.append(n_y[i].clone())
                a_s.append(s_a[i]); p_s.append(s_p[i]); n_s.append(s_n[i]); tp.append(ba[0])
                #print(ba[1].item(), type(ba[1].item()),  ba[1].item() == -1)
                if not ba[1].item() == -1:
                    # additional
                    x_a.append(a_x[i].clone()); x_p.append(n_x[i].clone()); x_n.append(p_x[i].clone())
                    y_a.append(a_y[i].clone()); y_p.append(n_y[i].clone()); y_n.append(p_y[i].clone())
                    a_s.append(s_a[i]); p_s.append(s_n[i]); n_s.append(s_p[i]); tp.append(ba[1])
            return (torch.stack(x_a, dim=0),
                    torch.stack(y_a, dim=0),
                    torch.stack(x_p, dim=0),
                    torch.stack(y_p, dim=0),
                    torch.stack(x_n, dim=0),
                    torch.stack(y_n, dim=0),
                    torch.stack(a_s, dim=0),
                    torch.stack(p_s, dim=0),
                    torch.stack(n_s, dim=0),
                    torch.stack(tp, dim=0))
        else:
            return a_x, a_y, p_x, p_y, n_x, n_y, s_a, s_p, s_n, triposi
    """

    def forward(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        #a_x, a_y, p_x, p_y, n_x, n_y, triposi, posiposi = self.dataload_tripet(batch)
        a_x_wave, a_y_wave, p_x_wave, p_y_wave, n_x_wave, n_y_wave, sound_a, sound_p, sound_n, triposi = batch
        # stft
        with torch.no_grad():
            a_x = self.transform_for_unet(a_x_wave); a_y = self.transform_for_unet(a_y_wave)
            p_x = self.transform_for_unet(p_x_wave); p_y = self.transform_for_unet(p_y_wave)
            n_x = self.transform_for_unet(n_x_wave); n_y = self.transform_for_unet(n_y_wave)
        #a_x, a_y, p_x, p_y, n_x, n_y, sound_a, sound_p, sound_n, triposi = self.clone_for_additional(a_x, a_y, p_x, p_y, n_x, n_y, sound_a, sound_p, sound_n, triposi)
        a_x["unet_out"] = self.unet(a_x["spec"])
        p_x["unet_out"] = self.unet(p_x["spec"])
        n_x["unet_out"] = self.unet(n_x["spec"])
        a_pred = self.transform_for_featurenet(**a_x)
        p_pred = self.transform_for_featurenet(**p_x)
        n_pred = self.transform_for_featurenet(**n_x)
        a_e, a_prob = self.featurenet(a_pred)
        p_e, p_prob = self.featurenet(p_pred)
        n_e, n_prob = self.featurenet(n_pred)
        # get loss
        loss_unet_a = self.get_loss_unet_triposi(a_y["spec"], a_x["unet_out"], triposi)
        loss_unet_p = self.get_loss_unet_triposi(p_y["spec"], p_x["unet_out"], triposi)
        loss_unet_n = self.get_loss_unet_triposi(n_y["spec"], n_x["unet_out"], triposi)
        loss_triplet_all, loss_triplet, dist_p, dist_n = self.get_loss_triplet(a_e, p_e, n_e, triposi)
        loss_recog = self.get_loss_recognise2(a_prob, sound_a) + self.get_loss_recognise2(p_prob, sound_p) + self.get_loss_recognise2(n_prob, sound_n)
        # record loss
        loss_all = (loss_unet_a + loss_unet_p + loss_unet_n)*self.cfg.unet_rate\
                    + loss_triplet_all*self.cfg.triplet_rate\
                    + loss_recog*self.cfg.recog_rate
        loss_unet = {
            "anchor": loss_unet_a.item(),
            "positive": loss_unet_p.item(),
            "negative": loss_unet_n.item(),
            "all": loss_unet_a.item() + loss_unet_p.item() + loss_unet_n.item()
        }
        loss_triplet["all"] = loss_triplet_all.item()
        prob = {inst: torch.concat([a_prob[inst], p_prob[inst], n_prob[inst]], dim=0) for inst in self.cfg.inst_list}
        cases = torch.concat([sound_a, sound_p, sound_n], dim=0)
        return loss_all, loss_unet, loss_triplet, dist_p, dist_n, loss_recog.item(), prob, cases
    
    def model_step(self, mode:str, batch):
        loss_all, loss_unet, loss_triplet, dist_p, dist_n, loss_recog, prob, cases = self.forward(batch)
        # update and log metrics
        self.recorder[mode]["loss_all"](loss_all)
        self.log(f"{mode}/loss_all", self.recorder[mode]["loss_all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # unet
        for type in self.song_type:
            self.recorder[mode]["loss_unet"][type](loss_unet[type])
            self.log(f"{mode}/loss_unet_{type}", self.recorder[mode]["loss_unet"][type], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        self.recorder[mode]["loss_unet"]["all"](loss_unet["all"])
        self.log(f"{mode}/loss_unet_all", self.recorder[mode]["loss_unet"]["all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # triplet
        for inst in self.cfg.inst_list:
            self.recorder[mode]["loss_triplet"][inst](loss_triplet[inst])
            self.recorder[mode]["dist_p"][inst](dist_p[inst])
            self.recorder[mode]["dist_n"][inst](dist_n[inst])
            self.log(f"{mode}/loss_triplet_{inst}", self.recorder[mode]["loss_triplet"][inst], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{mode}/dist_p_{inst}",       self.recorder[mode]["dist_p"][inst],       on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{mode}/dist_n_{inst}",       self.recorder[mode]["dist_n"][inst],       on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        self.recorder[mode]["loss_triplet"]["all"](loss_triplet["all"])
        self.log(f"{mode}/loss_triplet_all", self.recorder[mode]["loss_triplet"]["all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # recognize
        self.recorder[mode]["loss_recog"](loss_recog)
        self.log(f"{mode}/loss_recog", self.recorder[mode]["loss_recog"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        for idx,inst in enumerate(self.cfg.inst_list):
            self.recorder[mode]["recog_acc"][inst](prob[inst], cases[:,idx])
            self.log(f"{mode}/recog_acc_{inst}", self.recorder[mode]["recog_acc"][inst], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        # return loss or backpropagation will fail
        return loss_all
    
    """def model_step_psd(self, mode:str, batch, idx):
        ID, ver, seg, data_psd_wave, data_not_psd_wave, c = batch
        with torch.no_grad():
            data_psd = self.transform_for_unet(data_psd_wave)
            data_not_psd = self.transform_for_unet(data_not_psd_wave)
        data_psd["unet_out"] = self.unet(data_psd["spec"])
        data_not_psd["unet_out"] = self.unet(data_not_psd["spec"])
        pred_psd = self.transform_for_featurenet(**data_psd)
        pred_not_psd = self.transform_for_featurenet(**data_not_psd)
        embvec_psd, _ = self.featurenet(pred_psd)
        embvec_not_psd, _ = self.featurenet(pred_not_psd)
        #embvec, _, _ = self.net(data)
        if self.cfg.test_valid_norm:
            embvec_psd = torch.nn.functional.normalize(embvec_psd, dim=1)
            embvec_not_psd = torch.nn.functional.normalize(embvec_not_psd, dim=1)
        if len(self.cfg.inst_list) == 1:
            self.recorder_psd[mode]["label"][self.cfg.inst].append(torch.stack([ID, ver], dim=1))
            self.recorder_psd[mode]["vec_psd"][self.cfg.inst].append(embvec_psd)
            self.recorder_psd[mode]["vec_not_psd"][self.cfg.inst].append(embvec_not_psd)
        else:
            csn_valid = ConditionalSimNet1d().to(embvec_psd.device)
            self.recorder_psd[mode]["label"][self.cfg.inst_list[idx]].append(torch.stack([ID, ver], dim=1))
            self.recorder_psd[mode]["vec_psd"][self.cfg.inst_list[idx]].append(csn_valid(embvec_psd, c))
            self.recorder_psd[mode]["vec_not_psd"][self.cfg.inst_list[idx]].append(csn_valid(embvec_not_psd, c))
        if self.n_sound < 5:
            for inst in self.cfg.inst_list:
                if self.cfg.complex_unet:
                    z_psd = self.stft.spec2complex(data_psd["unet_out"][inst][5])
                    z_not_psd = self.stft.spec2complex(data_not_psd["unet_out"][inst][5])
                else:
                    z_psd = data_psd["unet_out"][inst][0] * data_psd["phase"][0]
                    z_not_psd = data_not_psd["unet_out"][inst][0] * data_not_psd["phase"][0]
                sound_psd = self.stft.istft(z_psd)
                sound_not_psd = self.stft.istft(z_not_psd)
                path = self.cfg.output_dir+f"/sound/{inst}/valid_e={self.current_epoch}"
                file_exist(path + "/psd")
                soundfile.write(path + f"/psd/separate{self.n_sound}_{inst}.wav", np.squeeze(sound_psd.to("cpu").numpy()), self.cfg.sr)
                file_exist(path + "/not_psd")
                soundfile.write(path + f"/not_psd/separate{self.n_sound}_{inst}.wav", np.squeeze(sound_not_psd.to("cpu").numpy()), self.cfg.sr)
                soundfile.write(path + f"/psd/mix{self.n_sound}_{inst}.wav", np.squeeze(data_psd_wave[0].to("cpu").numpy()), self.cfg.sr)
                soundfile.write(path + f"/not_psd/mix{self.n_sound}_{inst}.wav", np.squeeze(data_not_psd_wave[0].to("cpu").numpy()), self.cfg.sr)
                self.n_sound += 1"""
    
    def model_step_knn_tsne(self, mode:str, batch, idx, psd: str):
        ID, ver, seg, data_wave, c = batch
        with torch.no_grad():
            data = self.transform_for_unet(data_wave)
        data["unet_out"] = self.unet(data["spec"])
        pred = self.transform_for_featurenet(**data)
        embvec, _ = self.featurenet(pred)
        if self.cfg.test_valid_norm:
            embvec = torch.nn.functional.normalize(embvec, dim=1)
        if len(self.cfg.inst_list) == 1:
            self.recorder_psd[mode]["label"][psd][self.cfg.inst].append(torch.stack([ID, ver], dim=1))
            self.recorder_psd[mode]["vec"][psd][self.cfg.inst].append(embvec)
        else:
            csn_valid = ConditionalSimNet1d().to(embvec.device)
            self.recorder_psd[mode]["label"][psd][self.cfg.inst_list[idx]].append(torch.stack([ID, ver], dim=1))
            self.recorder_psd[mode]["vec"][psd][self.cfg.inst_list[idx]].append(csn_valid(embvec, c))
        if self.n_sound[mode][psd][self.cfg.inst_list[idx]] < 5:
            for inst in self.cfg.inst_list:
                if self.cfg.complex_unet:
                    z = self.stft.spec2complex(data["unet_out"][inst][5])
                else:
                    z = data["unet_out"][inst][0] * data["phase"][0]
                sound = self.stft.istft(z)
                path = self.cfg.output_dir+f"/sound/{inst}/{mode}_e={self.current_epoch}/{psd}"
                file_exist(path)
                soundfile.write(path + f"/separate{self.n_sound[mode][psd][self.cfg.inst_list[idx]]}_{inst}.wav", np.squeeze(sound.to("cpu").numpy()), self.cfg.sr)
                soundfile.write(path + f"/mix{self.n_sound[mode][psd][self.cfg.inst_list[idx]]}_{inst}.wav", np.squeeze(data_wave[0].to("cpu").numpy()), self.cfg.sr)
                self.n_sound[mode][psd][self.cfg.inst_list[idx]] += 1


    def training_step(
        self, batch, batch_idx: int
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_all = self.model_step("Train", batch)
        return loss_all

    def print_loss(self, mode:str):
        # unet
        print("\n\n== U-Net Loss ==")
        loss_unet = {type: self.recorder[mode]["loss_unet"][type].compute() for type in self.song_type}
        print(f"{mode} average loss UNet (anchor, positive, negative)  : {loss_unet['anchor']:2f}, {loss_unet['positive']:2f}, {loss_unet['negative']:2f}")
        loss_unet_all = self.recorder[mode]["loss_unet"]["all"].compute()
        print(f"{mode} average loss UNet            (all)              : {loss_unet_all:2f}")
        # triplet
        print("\n== Triplet Loss ==")
        for inst in self.cfg.inst_list:
            loss_triplet = self.recorder[mode]["loss_triplet"][inst].compute()
            dist_p = self.recorder[mode]["dist_p"][inst].compute()
            dist_n = self.recorder[mode]["dist_n"][inst].compute()
            print(f"{mode} average loss {inst:9}(Triplet, dist_p, dist_n) : {loss_triplet:2f}, {dist_p:2f}, {dist_n:2f}")
        loss_triplet_all = self.recorder[mode]["loss_triplet"]["all"].compute()
        print(f"{mode} average loss all      (Triplet)                 : {loss_triplet_all:2f}")
        # recognize
        print("\n== Recognize ==")
        print(f"{mode} average loss Recognize     : {self.recorder[mode]['loss_recog'].compute():2f}")
        for inst in self.cfg.inst_list:
            recog_acc = self.recorder[mode]["recog_acc"][inst].compute()
            print(f"{mode} average accuracy {inst:9} : {recog_acc*100:2f} %")
        print(f"\n== {mode} average loss all : {self.recorder[mode]['loss_all'].compute():2f}\n")
    
    def knn_tsne(self, mode:str, psd: str):
        print(f"\n== {psd} ==")
        acc_all = 0
        for inst in self.cfg.inst_list:
            label = torch.concat(self.recorder_psd[mode]["label"][psd][inst], dim=0).to("cpu").numpy()
            vec   = torch.concat(self.recorder_psd[mode]["vec"][psd][inst], dim=0).to("cpu").numpy()
            acc = knn_psd(label, vec, self.cfg, psd=False if psd == "not_psd" else True) # knn
            self.log(f"{mode}/knn_{psd}_{inst}", acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            if psd == "psd_mine":
                tsne_psd_marker(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            elif psd == "psd":
                tsne_psd(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            elif psd == "not_psd":
                tsne_not_psd(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            print(f"{mode} knn accuracy {inst:<10} {psd:<8} : {acc*100}%")
            acc_all += acc
        self.log(f"{mode}/knn_{psd}_avr", acc_all/len(self.cfg.inst_list), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        print(f"\n{mode} knn accuracy average {psd:<8}   : {acc_all/len(self.cfg.inst_list)*100}%")
        self.recorder_psd[mode]["label"][psd] = {inst:[] for inst in self.cfg.inst_list}; self.recorder_psd[mode]["vec"][psd] = {inst:[] for inst in self.cfg.inst_list}

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.print_loss("Train")

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        n_inst = len(self.cfg.inst_list)
        if dataloader_idx == 0:
            loss_all = self.model_step("Valid", batch)
        elif dataloader_idx > 0 and dataloader_idx < n_inst + 1:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx-1, psd="psd")
        elif dataloader_idx >= n_inst + 1 and dataloader_idx < 2*n_inst + 1:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx - 1 - n_inst, psd="not_psd")
        elif dataloader_idx >= 2*n_inst + 1 and dataloader_idx < 3*n_inst + 1:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx - 1 - 2*n_inst, psd="psd_mine")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.print_loss("Valid")
        self.knn_tsne("Valid", psd="psd")
        self.knn_tsne("Valid", psd="not_psd")
        self.knn_tsne("Valid", psd="psd_mine")
        for psd in ["psd", "not_psd", "psd_mine"]:
            self.n_sound["Valid"][psd] = {inst: 0 for inst in self.cfg.inst_list}

    def test_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        n_inst = len(self.cfg.inst_list)
        if dataloader_idx == 0:
            """cases_x, cases_y, cases = batch
            if self.cfg.complex:
                cases_x = self.stft.transform(cases_x)
                phase = None
            else:
                cases_x, phase = self.stft.transform(cases_x)
            c_e, c_prob, c_pred = self.net(cases_x)
            if not self.cfg.mel:
                for idx,c in enumerate(cases):
                    if ((c[0] == 1 and c[1] == 1 and c[2] == 1 and c[3] == 1 and c[4] == 1)
                        and self.n_sound < 5):
                        #sound = self.stft.detransform(cases_x[idx], phase[idx], param[0,idx], param[1,idx])
                        # TODO: complex = TrueだとphaseがNoneでidxがないって怒られるからそこを直す！いっそモデルの中で波にしちゃうのあり？てかそのロス追加する？
                        if self.cfg.complex:
                            sound = self.stft.detransform(cases_x[idx])
                        else:
                            sound = self.stft.detransform(cases_x[idx], phase[idx])
                        #sound = self.stft.detransform(cases_x[idx], phase[idx])
                        path = self.cfg.output_dir+f"/sound/mix"
                        file_exist(path)
                        soundfile.write(path + f"/separate{self.n_sound}_mix.wav", np.squeeze(sound.to("cpu").numpy()), self.cfg.sr)
                        for j,inst in enumerate(self.cfg.inst_list):
                            #sound = self.stft.detransform(cases_x[idx]*c_mask[inst][idx], phase[idx], param[0,idx], param[1,idx])
                            if self.cfg.complex:
                                sound = self.stft.detransform(c_pred[inst][idx])
                            else:
                                sound = self.stft.detransform(c_pred[inst][idx], phase[idx])
                            path = self.cfg.output_dir+f"/sound/{inst}"
                            file_exist(path)
                            soundfile.write(path + f"/separate{self.n_sound}_{inst}.wav", np.squeeze(sound.to("cpu").numpy()), self.cfg.sr)
                        self.n_sound += 1
            for idx,inst in enumerate(self.cfg.inst_list):
                self.recorder["Test"]["recog_acc"][inst](c_prob[inst], cases[:,idx])"""
            pass
        elif dataloader_idx > 0 and dataloader_idx < n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx-1, psd="psd")
        elif dataloader_idx >= n_inst + 1 and dataloader_idx < 2*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - n_inst, psd="not_psd")
        elif dataloader_idx >= 2*n_inst + 1 and dataloader_idx < 3*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - 2*n_inst, psd="psd_mine")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        print()
        """for inst in self.cfg.inst_list:
            recog_acc = self.recorder["Test"]["recog_acc"][inst].compute()
            print(f"Test average accuracy {inst:9} : {recog_acc*100: 2f} %")"""
        self.knn_tsne("Test", psd="psd")
        self.knn_tsne("Test", psd="not_psd")
        self.knn_tsne("Test", psd="psd_mine")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #    self.net = torch.compile(self.net)
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_all",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        
        #return torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg.lr)


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
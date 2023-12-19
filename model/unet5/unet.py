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
import museval

from utils.func import file_exist, knn_psd, tsne_psd, istft, tsne_psd_marker, TorchSTFT, evaluate
from ..csn import ConditionalSimNet1d
from ..tripletnet import CS_Tripletnet


class PL_UNet(LightningModule):
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
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cfg,
        ckpt_model_path,
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
        self.net = net
        model_checkpoint = {}
        if ckpt_model_path is not None:
            print("== Loading pretrained model...")
            checkpoint = torch.load(ckpt_model_path)
            for key in checkpoint["state_dict"]:
                model_checkpoint[key.replace("net.","")] = checkpoint["state_dict"][key]
            self.net.load_state_dict(model_checkpoint)
            print("== pretrained model was loaded!")
        print(net)

        # loss function
        self.loss_unet  = nn.L1Loss(reduction="mean")
        #self.loss_triplet = nn.MarginRankingLoss(margin=cfg.margin, reduction="none") #バッチ平均
        #self.loss_l2      = nn.MSELoss(reduction="sum")
        #self.loss_mrl     = nn.MarginRankingLoss(margin=cfg.margin, reduction="mean") #バッチ平均
        #self.loss_cross_entropy = nn.BCEWithLogitsLoss(reduction="mean")

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
        for step in ["Train", "Valid", "Test"]:
            self.recorder[step] = MDict({})
            self.recorder[step]["loss_all"] = MeanMetric()
            self.recorder[step]["loss_unet"] = MeanMetric()
            if step == "Test":
                self.recorder[step]["SDR"] = MeanMetric()
                self.recorder[step]["ISR"] = MeanMetric()
                self.recorder[step]["SIR"] = MeanMetric()
                self.recorder[step]["SAR"] = MeanMetric()
        self.n_sound = 0

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
        #for inst in self.cfg.inst_list:
        #    self.recorder["Valid"]["loss_triplet"][inst].reset()
        #    self.recorder["Valid"]["loss_recog"][inst].reset()
        #    self.recorder["Valid"]["recog_acc"][inst].reset()
        #    self.recorder["Valid"]["dist_p"][inst].reset()
        #    self.recorder["Valid"]["dist_n"][inst].reset()
        #self.recorder["Valid"]["loss_triplet"]["all"].reset()
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
    
    def get_loss_unet(self, pred, y):
        for idx, inst in enumerate(self.cfg.inst_list):
            loss = self.loss_unet(pred[inst], y[:, idx])
        return loss

    """
    def get_loss_unet_triposi(self, X, y, pred, triposi):
        # triplet positionのところのみ分離ロスを計算
        batch = X.shape[0]
        loss = 0
        for idx, c in enumerate(triposi): #個別音源でロスを計算
            loss += self.loss_unet(pred[self.cfg.inst_list[c.item()]][idx], y[idx, c])
        return loss / batch

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

    def get_loss_triplet(self, e_a, e_p, e_n, triposi):
        #batch = triposi.shape[0]
        tnet = CS_Tripletnet(ConditionalSimNet1d().to(e_a.device))
        distp, distn = tnet(e_a, e_p, e_n, triposi)
        target = torch.FloatTensor(distp.size()).fill_(1).to(distp.device) # 1で埋める
        loss = self.loss_triplet(distn, distp, target) # トリプレットロス
        loss_all = torch.sum(loss)/len(triposi)
        loss_inst  = {inst: torch.sum(loss[torch.where(triposi==i)])/len(torch.where(triposi==i)[0])  if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_list)}
        dist_p_all = {inst: torch.sum(distp[torch.where(triposi==i)])/len(torch.where(triposi==i)[0]) if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_list)}
        dist_n_all = {inst: torch.sum(distn[torch.where(triposi==i)])/len(torch.where(triposi==i)[0]) if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_list)}
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
    """
    
    def transform(self, x_wave, y_wave):
        device = x_wave.device
        if self.cfg.complex_unet:
            x = self.stft.transform(x_wave); y = self.stft.transform(y_wave)
        else:
            x, _ = self.stft.transform(x_wave); y, _ = self.stft.transform(y_wave)
        return x, y
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
    def forward(self, batch, mode):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        #a_x, a_y, p_x, p_y, n_x, n_y, triposi, posiposi = self.dataload_tripet(batch)
        mix_wave, stems_wave = batch
        # stft
        with torch.no_grad():
            if self.cfg.complex_unet:
                mix = self.stft.transform(mix_wave); stems = self.stft.transform(stems_wave); phase = None
            else:
                mix, phase = self.stft.transform(mix_wave); stems, _ = self.stft.transform(stems_wave)
        # forward
        pred = self.net(mix)
        # get loss
        loss_unet = self.get_loss_unet(pred, stems)
        # record loss
        loss_all = loss_unet
        if mode != "Train":
            pred_s = self.stft.detransform(pred) if phase is None else self.stft.detransform(pred, phase)
            return loss_all, loss_unet, pred_s
        else:
            return loss_all, loss_unet
    
    def model_step(self, mode:str, batch):
        if mode != "Train":
            loss_all, loss_unet, pred_s = self.forward(batch, mode)
            if self.n_sound < 5:
                path = self.cfg.output_dir+f"/sound/{self.cfg.inst}/valid_e={self.current_epoch}"
                file_exist(path)
                soundfile.write(path + f"/separate{self.n_sound}_{self.cfg.inst}.wav", torch.squeeze(pred_s).to("cpu").numpy(), self.cfg.sr)
        else:
            loss_all, loss_unet = self.forward(batch, mode)
        # update and log metrics
        self.recorder[mode]["loss_all"](loss_all)
        self.log(f"{mode}/loss_all", self.recorder[mode]["loss_all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # unet
        self.recorder[mode]["loss_unet"](loss_unet)
        self.log(f"{mode}/loss_unet", self.recorder[mode]["loss_unet"], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        #evaluate(reference=y[...,:pred_sound.shape[-1]].to("cpu"), estimate=pred_sound.to("cpu"), inst_list=self.inst_list, writer=self.writer, epoch=self.epoch)
        # return loss or backpropagation will fail
        return loss_all
    """
    def model_step_psd(self, mode:str, batch, idx):
        ID, ver, seg, data_wave, c = batch
        with torch.no_grad():
            if self.cfg.complex:
                data = self.stft.transform(data_wave)
            else:
                data, _ = self.stft.transform(data_wave)
        embvec, _, _ = self.net(data)
        if self.cfg.test_valid_norm:
            embvec = torch.nn.functional.normalize(embvec, dim=1)
        csn_valid = ConditionalSimNet1d().to(embvec.device)
        self.recorder_psd[mode]["label"][self.cfg.inst_list[idx]].append(torch.stack([ID, ver], dim=1))
        self.recorder_psd[mode]["vec"][self.cfg.inst_list[idx]].append(csn_valid(embvec, c))
    """
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
        loss_unet = self.recorder[mode]["loss_unet"].compute()
        print(f"{mode} average loss UNet {self.cfg.inst} : {loss_unet:2f}")
        print(f"\n== {mode} average loss all : {self.recorder[mode]['loss_all'].compute():2f}\n")
    
    """
    def knn_tsne(self, mode:str):
        acc_all = 0
        for inst in self.cfg.inst_list:
            label = torch.concat(self.recorder_psd[mode]["label"][inst], dim=0).to("cpu").numpy()
            vec   = torch.concat(self.recorder_psd[mode]["vec"][inst], dim=0).to("cpu").numpy()
            acc = knn_psd(label, vec, self.cfg) # knn
            self.log(f"{mode}/knn_{inst}", acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            if self.cfg.test_psd_mine:
                tsne_psd_marker(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}", current_epoch=self.current_epoch) # tsne
            else:
                tsne_psd(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}", current_epoch=self.current_epoch) # tsne
            print(f"{mode} knn accuracy {inst:<10}: {acc*100}%")
            acc_all += acc
        self.log(f"{mode}/knn_avr", acc_all/len(self.cfg.inst_list), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        print(f"\n{mode} knn accuracy average   : {acc_all/len(self.cfg.inst_list)*100}%")
        self.recorder_psd[mode]["label"] = {inst:[] for inst in self.cfg.inst_list}; self.recorder_psd[mode]["vec"] = {inst:[] for inst in self.cfg.inst_list}
    """

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.print_loss("Train")

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        #a_x, a_y, p_x, p_y, n_x, n_y, triposi, posiposi = self.dataload_tripet(batch)
        mix_wave, stems_wave = batch
        # stft
        with torch.no_grad():
            if self.cfg.complex_unet:
                mix = self.stft.transform(mix_wave); stems = self.stft.transform(stems_wave); phase = None
            else:
                mix, phase = self.stft.transform(mix_wave); stems, _ = self.stft.transform(stems_wave)
        # forward
        pred = self.net(mix)
        # get loss
        loss_unet = self.get_loss_unet(pred, stems)
        # record loss
        loss_all = loss_unet
        # Save predicted sound.
        if self.n_sound < 5:
            for i, inst in enumerate(self.cfg.inst_list):
                pred_s = self.stft.detransform(pred[inst]) if phase is None else self.stft.detransform(pred[inst], phase)
                path = self.cfg.output_dir+f"/sound/{inst}/valid_e={self.current_epoch}"
                file_exist(path)
                idx = 0
                while self.n_sound < 5:
                    soundfile.write(path + f"/separate{self.n_sound}_{inst}.wav", torch.squeeze(pred_s[idx]).to("cpu").numpy(), self.cfg.sr)
                    idx += 1; self.n_sound += 1
        # update and log metrics
        self.recorder["Valid"]["loss_all"](loss_all)
        self.log(f"Valid/loss_all", self.recorder["Valid"]["loss_all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # unet
        self.recorder["Valid"]["loss_unet"](loss_unet)
        self.log(f"Valid/loss_unet", self.recorder["Valid"]["loss_unet"], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        #evaluate(reference=y[...,:pred_sound.shape[-1]].to("cpu"), estimate=pred_sound.to("cpu"), inst_list=self.inst_list, writer=self.writer, epoch=self.epoch)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.n_sound = 0
        self.print_loss("Valid")

    def evaluate(self, reference, estimate):
        # assume mix as estimates
        if reference.shape[3] > estimate.shape[3]:
            reference = reference[..., :estimate.shape[3]]
        B, C, S, T = reference.shape
        reference = torch.reshape(reference, (B, T, C*S))
        estimate  = torch.reshape(estimate, (B, T, C*S))
        scores = {"SDR":0, "ISR":0, "SIR":0, "SAR":0}
        # Evaluate using museval
        score = museval.evaluate(references=reference.to("cpu"), estimates=estimate.to("cpu"))
        #print(score)
        for i,key in enumerate(list(scores.keys())):
            #print(score[i].shape)
            scores[key] = np.mean(score[i])
        return scores

    def test_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        #a_x, a_y, p_x, p_y, n_x, n_y, triposi, posiposi = self.dataload_tripet(batch)
        mix_wave, stems_wave = batch
        # stft
        with torch.no_grad():
            if self.cfg.complex_unet:
                mix = self.stft.transform(mix_wave); stems = self.stft.transform(stems_wave); phase = None
            else:
                mix, phase = self.stft.transform(mix_wave); stems, _ = self.stft.transform(stems_wave)
        # forward
        pred = self.net(mix)
        # get loss
        loss_unet = self.get_loss_unet(pred, stems)
        # record loss
        loss_all = loss_unet
        # Save predicted sound.
        pred_s = self.stft.detransform(pred[self.cfg.inst]) if phase is None else self.stft.detransform(pred[self.cfg.inst], phase)
        if self.n_sound < 10:
            path = self.cfg.output_dir+f"/sound/{self.cfg.inst}/test"
            file_exist(path)
            idx = 0
            while self.n_sound < 5:
                soundfile.write(path + f"/reference{self.n_sound}_{self.cfg.inst}.wav", torch.squeeze(stems_wave[idx]).to("cpu").numpy(), self.cfg.sr)
                soundfile.write(path + f"/separate{self.n_sound}_{self.cfg.inst}.wav", torch.squeeze(pred_s[idx]).to("cpu").numpy(), self.cfg.sr)
                idx += 1; self.n_sound += 1
        # update and log metrics
        self.recorder["Test"]["loss_all"](loss_all)
        self.log(f"Test/loss_all", self.recorder["Test"]["loss_all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # unet
        self.recorder["Test"]["loss_unet"](loss_unet)
        self.log(f"Test/loss_unet", self.recorder["Test"]["loss_unet"], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        # Evaluate predicted sound.
        pred_s = torch.unsqueeze(pred_s, dim=1) # B, C, S, TのC(楽器音数)をつくる。ちなみにSはmono or stereo
        #print(stems_wave.shape, pred_s.shape)
        scores = self.evaluate(reference=stems_wave, estimate=pred_s)
        for i,key in enumerate(list(scores.keys())):
            self.recorder["Test"][key](scores[key])
            self.log(f"Test/{key}", self.recorder["Test"][key], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        print()
        print(f'{self.cfg.inst:<10}- SDR: {self.recorder["Test"]["SDR"].compute():.3f}, ISR: {self.recorder["Test"]["ISR"].compute():.3f}, SIR: {self.recorder["Test"]["SIR"].compute():.3f}, SAR: {self.recorder["Test"]["SAR"].compute():.3f}')
        self.n_sound = 0

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
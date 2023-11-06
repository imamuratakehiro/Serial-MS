from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm

from utils.func import file_exist, knn_psd, tsne_psd, TorchSTFT
from ..jnet.model_jnet_128_embnet import JNet128Embnet
from ..csn import ConditionalSimNet1d


class PreTrain32(LightningModule):
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
        cfg) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        assert cfg.condition32, f"Please change the value condition32 to True. Now, it is {cfg.condition32}."

        # network
        self.net = net
        print(net)

        # loss function
        self.loss_triplet = nn.MarginRankingLoss(margin=cfg.margin, reduction="mean") #バッチ平均
        self.loss_mse = nn.MSELoss(reduction="mean")

        # metric objects for calculating and averaging accuracy across batches
        #self.train_acc = Accuracy(task="multiclass", num_classes=10)
        #self.val_acc = Accuracy(task="multiclass", num_classes=10)
        #self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss_mix  = MeanMetric()
        self.train_loss_inst = MeanMetric()
        #self.val_loss   = MeanMetric()
        #self.train_loss_inst = {inst: MeanMetric() for inst in cfg.inst_list}
        #self.train_dist_p = {inst: MeanMetric() for inst in cfg.inst_list}
        #self.train_dist_n = {inst: MeanMetric() for inst in cfg.inst_list}
        #self.val_loss_inst = {inst: MeanMetric() for inst in cfg.inst_list}
        #self.val_dist_p = {inst: MeanMetric() for inst in cfg.inst_list}
        #self.val_dist_n = {inst: MeanMetric() for inst in cfg.inst_list}
        
        # for valid and test
        self.valid_label = {inst:[] for inst in cfg.inst_list}; self.valid_vec = {inst:[] for inst in cfg.inst_list}
        self.test_label  = {inst:[] for inst in cfg.inst_list}; self.test_vec  = {inst:[] for inst in cfg.inst_list}
        #self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        #self.val_acc_best = MaxMetric()
        self.stft = TorchSTFT(cfg=cfg)
        self.cfg = cfg
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # sanity checkingでvalidを回している。なのでlossが追加されているから、リセットが必要！
        #self.val_loss.reset()
        #self.val_acc.reset()
        #self.val_acc_best.reset()
        pass
    
    def forward_mix(self, X):
        # 順伝播
        emb_mix, _, _ = self.net(X)
        return emb_mix
    
    def forward_inst(self, y):
        emb_insts = []
        for idx,inst in enumerate(self.cfg.inst_list):
            emb_inst, _, _ = self.net(y[:,idx])
            emb_insts.append(emb_inst)
        return torch.stack(emb_insts, dim=1)

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        mix, condition, emb_target = batch
        _, mix, _ = self.stft.transform(mix)
        emb_mix   = self.forward_mix(mix)
        # loss
        loss_mix = self.loss_mse(emb_mix, emb_target)
        #a_e = self.forward(a_x)
        #p_e = self.forward(p_x)
        #n_e = self.forward(n_x)
        #loss_triplet_all, loss_triplet, dist_p, dist_n = self.get_loss_triplet(a_e, p_e, n_e, triposi, posiposi)
        #loss = self.criterion(logits, y)
        #preds = torch.argmax(logits, dim=1)
        return loss_mix

    def training_step(
        self, batch, batch_idx: int
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_mix = self.model_step(batch)
        # update and log metrics
        self.train_loss_mix(loss_mix)
        self.log("train/loss", self.train_loss_mix, on_step=True, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss_mix

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        print(f"\nTrain average loss <input 32condition > : {self.train_loss_mix.compute()}")

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        ID, ver, seg, data, c = batch
        _, data, _ = self.stft.transform(data)
        embvec = self.forward_mix(data)
        # インスタンスを生成していなかったら、生成する。
        csn_valid = ConditionalSimNet1d()
        self.valid_label[self.cfg.inst_list[dataloader_idx]].append(torch.stack([ID, ver], dim=1))
        self.valid_vec[self.cfg.inst_list[dataloader_idx]].append(csn_valid(embvec, c))

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        print("knn")
        acc_all = 0
        for inst in self.cfg.inst_list:
            label = torch.concat(self.valid_label[inst], dim=0).to("cpu").numpy()
            vec   = torch.concat(self.valid_vec[inst], dim=0).to("cpu").numpy()
            acc = knn_psd(label, vec, self.cfg) # knn
            self.log(f"val/knn_{inst}", acc, on_step=False, on_epoch=True, prog_bar=False)
            tsne_psd(label, vec, self.cfg, dir_path=self.cfg.output_dir+f"/{inst}", current_epoch=self.current_epoch) # tsne
            print(f"knn accuracy valid {inst:<10}: {acc*100}%")
            acc_all += acc
        self.log(f"val/knn_avr", acc_all/len(self.cfg.inst_list), on_step=False, on_epoch=True, prog_bar=True)
        print(f"\nknn accuracy valid average   : {acc_all/len(self.cfg.inst_list)*100}%")
        self.valid_label = {inst:[] for inst in self.cfg.inst_list}; self.valid_vec = {inst:[] for inst in self.cfg.inst_list}

    def test_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        ID, ver, seg, data, c = batch
        _, data, _ = self.stft.transform(data)
        embvec = self.forward_mix(data)
        csn_test = ConditionalSimNet1d(data.shape[0]) # csnのモデルを保存されないようにするために配列に入れる
        self.test_label[self.cfg.inst_list[dataloader_idx]].append(torch.stack([ID, ver], dim=1))
        self.test_vec[self.cfg.inst_list[dataloader_idx]].append(csn_test(embvec, c))

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        acc_all = 0
        for inst in self.cfg.inst_list:
            label = torch.concat(self.test_label[inst], dim=0).to("cpu").numpy()
            vec   = torch.concat(self.test_vec[inst], dim=0).to("cpu").numpy()
            acc = knn_psd(label, vec, self.cfg) # knn
            tsne_psd(label, vec, self.cfg, dir_path=self.cfg.output_dir+f"/{inst}") # tsne
            self.log(f"test/knn_{inst}", acc, on_step=False, on_epoch=True, prog_bar=False)
            print(f"knn accuracy test {inst:<10}: {acc*100}%")
            acc_all += acc
        self.log(f"test/knn_avr", acc_all/len(self.cfg.inst_list), on_step=False, on_epoch=True, prog_bar=True)
        print(f"\nknn accuracy test average   : {acc_all/len(self.cfg.inst_list)*100}%")


    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        #if self.hparams.compile and stage == "fit":
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
                    "monitor": self.cfg.monitor_sch,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        
        #return torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg.lr)


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
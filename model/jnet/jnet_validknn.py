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

from utils.func import file_exist


class JNetValidKnn(LightningModule):
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

        # network
        self.net = net
        print(net)

        # loss function
        self.loss_triplet = nn.MarginRankingLoss(margin=cfg.margin, reduction="mean") #バッチ平均

        # metric objects for calculating and averaging accuracy across batches
        #self.train_acc = Accuracy(task="multiclass", num_classes=10)
        #self.val_acc = Accuracy(task="multiclass", num_classes=10)
        #self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.train_loss_triplet = {inst: MeanMetric() for inst in cfg.inst_list}
        self.train_dist_p = {inst: MeanMetric() for inst in cfg.inst_list}
        self.train_dist_n = {inst: MeanMetric() for inst in cfg.inst_list}
        self.val_loss_triplet = {inst: MeanMetric() for inst in cfg.inst_list}
        self.val_dist_p = {inst: MeanMetric() for inst in cfg.inst_list}
        self.val_dist_n = {inst: MeanMetric() for inst in cfg.inst_list}
        
        # for valid and test
        self.valid_label = []; self.valid_vec = []
        self.test_label  = []; self.test_vec  = []
        #self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        #self.val_acc_best = MaxMetric()
        
        self.cfg = cfg
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        #self.val_acc.reset()
        #self.val_acc_best.reset()
        
    def dataload(self, batch):
        # データセットを学習部分と教師部分に分けてdeviceを設定
        #self.logger.s_dataload()
        #print(f"\t..................dataset log.......................")
        anchor_X, positive_X, negative_X, triposi, posiposi = batch
        #print(f"\t....................................................")
        #self.logger.f_dataload()
        return anchor_X, positive_X, negative_X, triposi, posiposi
    
    def get_loss_triplet(self, masked_embedded_a, masked_embedded_p, masked_embedded_n, triposi, posiposi, l=1):
        batch = posiposi.shape[0]
        loss_all = 0
        loss = {inst: 0 for inst in self.cfg.inst_list}
        dist_p_all = {inst: 0 for inst in self.cfg.inst_list}
        dist_n_all = {inst: 0 for inst in self.cfg.inst_list}
        for b in range(batch):
            for i in triposi[b]:
                i = i.item()
                condition = self.cfg.inst_list[i]
                if posiposi[b, i] == 1:
                    dist_p = F.pairwise_distance(masked_embedded_a[condition][b], masked_embedded_p[condition][b], 2)
                    dist_n = F.pairwise_distance(masked_embedded_a[condition][b], masked_embedded_n[condition][b], 2)
                    #dist_p = self.cos_sim(masked_embedded_a[condition][i], masked_embedded_p[condition][i])
                    #dist_n = self.cos_sim(masked_embedded_a[condition][i], masked_embedded_n[condition][i])
                elif posiposi[b, i] == 2:
                    # 距離を計算、positiveとnegativeを逆に
                    dist_n = F.pairwise_distance(masked_embedded_a[condition][b], masked_embedded_p[condition][b], 2)
                    dist_p = F.pairwise_distance(masked_embedded_a[condition][b], masked_embedded_n[condition][b], 2)
                target = torch.ones_like(dist_p).to(dist_p.device) #1で埋める
                # トリプレットロス
                triplet = self.loss_triplet(dist_n, dist_p, target) * l #3つのshapeを同じにする
                loss[condition] += triplet.item() / batch
                loss_all += triplet
                dist_p_all[condition] += dist_p.item() / batch
                dist_n_all[condition] += dist_n.item() / batch
        return loss_all, loss, dist_p_all, dist_n_all
    
    def forward(self, X):
        # 順伝播
        embedded_vec, _ = self.net(X)
        return embedded_vec

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        a_x, p_x, n_x, triposi, posiposi = self.dataload(batch)
        a_e = self.forward(a_x)
        p_e = self.forward(p_x)
        n_e = self.forward(n_x)
        loss_triplet_all, loss_triplet, dist_p, dist_n = self.get_loss_triplet(a_e, p_e, n_e, triposi, posiposi)
        #loss = self.criterion(logits, y)
        #preds = torch.argmax(logits, dim=1)
        return loss_triplet_all, loss_triplet, dist_p, dist_n

    def training_step(
        self, batch, batch_idx: int
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_triplet_all, loss_triplet, dist_p, dist_n = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss_triplet_all)
        self.log("train/loss_all", self.train_loss, on_step=True, on_epoch=True, prog_bar=False)
        for inst in self.cfg.inst_list:
            self.train_loss_triplet[inst] = loss_triplet[inst]
            self.train_dist_p[inst] = dist_p[inst]
            self.train_dist_n[inst] = dist_n[inst]
            self.log(f"train/loss_{inst}", self.train_loss_triplet[inst], on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"train/dist_p_{inst}", self.train_dist_p[inst], on_step=True, on_epoch=True, prog_bar=True)
            self.log(f"train/dist_n_{inst}", self.train_dist_n[inst], on_step=True, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss_triplet_all

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        label, X = batch
        e = self.forward(X)
        self.valid_label.append(label.detach())
        self.valid_vec.append(e[self.cfg.inst].detach())

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        #acc = self.val_acc.compute()  # get current val acc
        #self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        #self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        # knn
        print("knn")
        label = torch.concat(self.valid_label, dim=0).to("cpu").numpy()
        vec   = torch.concat(self.valid_vec, dim=0).to("cpu").numpy()
        total_all   = {inst: 0 for inst in self.cfg.inst_list}
        correct_all = {inst: 0 for inst in self.cfg.inst_list}
        for idx in range(len(label)):
            knn_sk = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=self.cfg.num_workers)
            knn_sk.fit(np.delete(vec, idx, 0), np.delete(label, idx, 0))
            pred = knn_sk.predict(vec[idx].reshape(1, -1))
            #print(label[idx], pred)
            #print(f"{inst:<10}: {metrics.accuracy_score([info_list[idx]], pred)}%")
            if label[idx] == pred:
                correct_all[self.cfg.inst] += 1
            total_all[self.cfg.inst] += 1
        #print(f"{inst:<10}: {correct_all[inst]/total_all[inst]}%")
        self.log(f"val/knn", correct_all[self.cfg.inst]/total_all[self.cfg.inst], on_step=False, on_epoch=True, prog_bar=True)
        self.valid_label = []; self.valid_vec = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        label, X = batch
        e = self.forward(X)
        self.test_label.append(label)
        self.test_vec.append(e[self.cfg.inst])

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # knn
        print("knn")
        label = torch.concat(self.test_label, dim=0).to("cpu").numpy()
        vec   = torch.concat(self.test_vec, dim=0).to("cpu").numpy()
        total_all   = {inst: 0 for inst in self.cfg.inst_list}
        correct_all = {inst: 0 for inst in self.cfg.inst_list}
        for idx in range(len(label)):
            knn_sk = KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=self.cfg.num_workers)
            knn_sk.fit(np.delete(vec, idx, 0), np.delete(label, idx, 0))
            pred = knn_sk.predict(vec[idx].reshape(1, -1))
            #print(label[idx], pred)
            #print(f"{inst:<10}: {metrics.accuracy_score([info_list[idx]], pred)}%")
            if label[idx] == pred:
                correct_all[self.cfg.inst] += 1
            total_all[self.cfg.inst] += 1
        for idx, inst in enumerate(self.cfg.inst_list):
            print(f"{inst:<10}: {correct_all[inst]/total_all[inst]*100}%")
            self.log(f"test/knn", correct_all[inst]/total_all[inst], on_step=False, on_epoch=True, prog_bar=True, logger=False)

        # tsne
        counter = 0
        cmap = plt.cm.get_cmap("tab20")
        label20 = []
        color20 = []
        label_picked = []
        vec20 = []
        while counter < 20:
            pick_label = np.random.choice(label)
            if pick_label in label_picked:
                continue
            samesong_idx = np.where(label==pick_label)[0]
            #label20.append(label[samesong_idx])
            color20 = color20 + [cmap.colors[counter] for i in range(samesong_idx.shape[0])]
            vec20.append(vec[samesong_idx])
            label_picked.append(pick_label)
            counter += 1
        #color20 = np.concatenate(color20, axis=0)
        vec20 = np.concatenate(vec20, axis=0)
        perplexity = [5, 15, 30, 50]
        for i in range(len(perplexity)):
            fig, ax = plt.subplots(1, 1)
            X_reduced = TSNE(n_components=2, random_state=0, perplexity=perplexity[i]).fit_transform(vec20)
            mappable = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color20, s=30)
            #fig.colorbar(mappable, norm=BoundaryNorm(bounds,cmap.N))
            dir_path = self.cfg.output_dir
            file_exist(dir_path)
            fig.savefig(dir_path + f"/emb_s{20}_tsne_p{perplexity[i]}_m{self.cfg.margin}.png")
            plt.clf()
            plt.close()

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
                    "monitor": "val/knn",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        
        #return torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg.lr)


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
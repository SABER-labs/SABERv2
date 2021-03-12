import pytorch_lightning as pl
import torch
import math
from model.quartznet import QuartzNet
from utils.config import config
from utils.training_utils import get_adam_warmup_cosine_schedule, length_to_mask
from losses.contrastive_loss import nt_xent_loss
import torch.nn.functional as F
from model.projection_head import Projection
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer
from typing import Callable, Optional

class SpeechSimClr(pl.LightningModule):

    def __init__(self, num_train_samples):
        super().__init__()
        self.encoder = QuartzNet(n_mels=config.audio.n_mels)
        global_batch_size = config.dataloader.batch_size * config.trainer.num_gpus
        self.lr_schedule = get_adam_warmup_cosine_schedule(num_train_samples // global_batch_size)
        self.projection = Projection()
        self.model_stride = self.encoder.model_stride()
        self.criterion = NT_Xent(config.dataloader.batch_size, config.simclr.temperature, config.trainer.num_gpus * config.trainer.num_nodes)

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        (img1, img2, img1_len, img2_len) = batch

        h1_mask = length_to_mask(img1_len, stride=self.model_stride, max_len=img1.size(2))
        h2_mask = length_to_mask(img2_len, stride=self.model_stride, max_len=img2.size(2))

        # get h representations, bolts resnet returns a list
        h1 = self(img1) * h1_mask[:, None, :]
        h2 = self(img2) * h2_mask[:, None, :]

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.criterion(z1, z2)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('learning_rate', self.lr_schedule[self.trainer.global_step], on_step=True, on_epoch=False)
        self.log('simclr_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('simclr_loss_epoch', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.trainer.learning_rate, weight_decay=config.trainer.weight_decay)
        return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        # from lightning
        if not isinstance(optimizer, LightningOptimizer):
            # wraps into LightingOptimizer only for running step
            optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer)
        optimizer.step(closure=optimizer_closure)
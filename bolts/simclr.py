import pytorch_lightning as pl
import torch
import math
from model.quartznet import QuartzNet
from utils.config import config
from utils.training_utils import length_to_mask
from losses.contrastive_loss import NT_Xent
import torch.nn.functional as F
from model.projection_head import Projection
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

class SpeechSimClr(pl.LightningModule):

    def __init__(self, num_train_samples):
        super().__init__()
        self.encoder = QuartzNet(n_mels=config.audio.n_mels)
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

        return self.criterion(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.trainer.learning_rate, weight_decay=config.trainer.weight_decay)
        optimizer = LARSWrapper(optimizer)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=config.trainer.warmup_epochs,
            max_epochs=config.trainer.max_epochs,
            warmup_start_lr=config.trainer.start_lr,
            eta_min=config.trainer.final_lr
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
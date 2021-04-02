import pytorch_lightning as pl
import torch
import math
from model.quartznet import QuartzNet
from utils.config import config
from utils.training_utils import length_to_mask
import torch.nn.functional as F
from model.projection_head import SupervisedHead
from pl_bolts.optimizers.lars_scheduling import LARSWrapper
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from jiwer import wer
import sentencepiece as spm

class SupervisedTask(pl.LightningModule):

    def __init__(self, num_samples):
        super().__init__()
        self.encoder = QuartzNet(n_mels=config.audio.n_mels)
        self.steps_per_epoch = (num_samples // (config.dataloader.batch_size * config.trainer.num_gpus * config.trainer.num_nodes)) + 1
        self.projection = SupervisedHead()
        self.model_stride = self.encoder.model_stride() 
        self.criterion = torch.nn.CTCLoss()
        spe = spm.SentencePieceProcessor()
        self.tokenizer = spe.load(config.supervised_train.language_modelpath)
        self.load_encoder_params(config.supervised_train.encoder_weights_path)

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        return F.log_softmax(x, dim=-1)

    def shared_step(self, batch):
        (img1, img1_len, target, target_len) = batch
        h = self(img1)
        h = h.permute(1, 0, 2)
        return self.criterion(h, target, img1_len, target_len)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):

        (img1, img1_len, target, target_len) = batch
        h = self(img1)
        h = h.permute(1, 0, 2)
        loss = self.criterion(h, target, img1_len, target_len)
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        h = torch.argmax(h.permute(1, 0, 2), 2)
        h = h.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        ref = self.tokenizer.decode(h)
        pred = self.tokenizer.decode(target)

        error = wer(ref, pred)
        self.log("wer", error, on_step=True, on_epoch=False)

        return loss
        

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.trainer.learning_rate, weight_decay=config.trainer.weight_decay)
        optimizer = LARSWrapper(optimizer, eta=0.001, clip=False)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=int(config.trainer.warmup_epochs * self.steps_per_epoch),
            max_epochs=int(config.trainer.max_epochs * self.steps_per_epoch),
            warmup_start_lr=config.trainer.start_lr,
            eta_min=config.trainer.final_lr
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def load_encoder_params(self, path):
        state_dict = torch.load(path)
        model_state_dict = self.encoder.state_dict()

        final_dict = dict()
        for k in state_dict:
            if k in model_state_dict and state_dict[k].shape == model_state_dict[k].shape:                
                final_dict[k] = state_dict[k]

        self.encoder.load_state_dict(final_dict)


    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

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
from bolts.decoder import GreedyDecoder

class SupervisedTask(pl.LightningModule):

    def __init__(self, num_samples):
        super().__init__()
        self.encoder = QuartzNet(n_mels=config.audio.n_mels)
        self.steps_per_epoch = (num_samples // (config.dataloader.batch_size * config.trainer.num_gpus * config.trainer.num_nodes)) + 1
        self.projection = SupervisedHead()
        self.model_stride = self.encoder.model_stride() 
        self.criterion = torch.nn.CTCLoss(blank=config.supervised_train.nclasses-1,zero_infinity=True)
        blank_label = config.supervised_train.nclasses -1
        self.decoder = GreedyDecoder(blank_label)
        # self.load_encoder_params(config.supervised_train.encoder_weights_path)

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
        
        # self.log('val_loss', loss, on_step=True, on_epoch=False)
        # print ('val_loss', loss)
        h = h.permute(1, 0, 2).cpu().numpy()
        target = target.cpu().numpy()
        out, _ = self.decoder.decode(h) 

        wers = []
        cers = []
        for i in len(target):
            wers.append(self.decoder.wer(out[i], target[i]))
            cers.append(self.decoder.cer(out[i], target[i]))

        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('wer', sum(wers)/len(wers), on_step=True, on_epoch=True, sync_dist=True)
        self.log('cer', sum(cers)/len(cers), on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
        
    def validation_epoch_end(self, output):
        # result = pl.EvalResult(checkpoint_on=output['valid_loss'].mean())
        # print(torch.stack([x['val_loss'] for x in output]).mean())
        avg_val_loss = torch.stack([x['val_loss'] for x in output]).mean()
        error = [x['wer'] for x in output]
        avg_error = sum(error)/len(error)
        self.log('avg_val_loss', avg_val_loss, logger=True)
        self.log('wer', avg_error)

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
        state_dict = torch.load(path)['state_dict']
        model_state_dict = self.encoder.state_dict()

        final_dict = dict()
        encoder_name = "encoder."
        for k in model_state_dict:
            model_key = encoder_name + k
            if model_key in state_dict and state_dict[model_key].shape == model_state_dict[k].shape:                
                final_dict[k] = state_dict[model_key]

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

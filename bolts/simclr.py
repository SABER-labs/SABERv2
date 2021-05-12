import pytorch_lightning as pl
import torch
from model.streaming_convnets import Streaming_convnets
from utils.config import config
from utils.training_utils import length_to_mask
from losses.contrastive_loss import NT_Xent
from model.projection_head import Projection
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class SpeechSimClr(pl.LightningModule):

    def __init__(self, num_samples):
        super().__init__()
        self.encoder = Streaming_convnets(0.1, config.audio.n_mels, 1)
        self.steps_per_epoch = (num_samples // (config.dataloader.batch_size *
                                config.trainer.num_gpus * config.trainer.num_nodes)) + 1
        self.projection = Projection()
        self.model_stride = self.encoder.model_stride()
        self.criterion = NT_Xent(
            config.dataloader.batch_size,
            config.simclr.temperature,
            config.trainer.num_gpus * config.trainer.num_nodes,
            config.simclr.margin
        )

    def forward(self, x):
        return self.encoder(x.unsqueeze(1))

    def shared_step(self, batch):
        (img1, img2, img1_len, img2_len) = batch

        h1_mask = length_to_mask(
            img1_len, stride=self.model_stride, max_len=img1.size(2))
        h2_mask = length_to_mask(
            img2_len, stride=self.model_stride, max_len=img2.size(2))

        # get h representations, bolts resnet returns a list
        h1 = self(img1)
        h2 = self(img2)

        h1 *= h1_mask[:, None, :h1.size(2)]
        h2 *= h2_mask[:, None, :h2.size(2)]

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        return self.criterion(z1, z2)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def test_step(self, batch, batch_idx):
        (img, _, img_len, _) = batch
        h_mask = length_to_mask(
            img_len, stride=self.model_stride, max_len=img.size(2))
        h = self(img) * h_mask[:, None, :]
        z = self.projection(h)
        similarity_f = torch.nn.CosineSimilarity(dim=2)
        scores = similarity_f(z.unsqueeze(1), z.unsqueeze(0))
        self.print("\n")
        self.print(scores)
        self.print(
            f"Account Balance positive 1 - arjun x mohit = {scores[0][3]}")
        self.print(f"Sentence positive 1 - arjun x mohit = {scores[2][4]}")
        self.print(f"Sentence positive 2 - arjun x souvik = {scores[2][5]}")
        self.print(f"Sentence positive 3 - mohit x souvik = {scores[4][5]}")
        self.print(f"Easy Negative 1 - mohit x souvik = {scores[3][5]}")
        self.print(f"Easy Negative 2 - arjun x souvik = {scores[0][5]}")
        self.print(f"Easy Negative 3 - mohit x arjun = {scores[3][2]}")
        self.print(f"Easy Negative 4 - arjun x mohit = {scores[0][4]}")
        self.print(
            f"Sentence hard negative 1 - arjun x arjun = {scores[1][2]}")
        self.print(
            f"Sentence hard negative 2 - arjun x mohit = {scores[1][4]}")
        self.print(
            f"Sentence hard negative 3 - arjun x souvik = {scores[1][5]}")

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=int(config.trainer.warmup_epochs *
                              self.steps_per_epoch),
            max_epochs=int(config.trainer.max_epochs * self.steps_per_epoch),
            warmup_start_lr=config.trainer.start_lr,
            eta_min=config.trainer.final_lr
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

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

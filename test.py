import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.config import config
from bolts.unsupervised_data import UnsupervisedCommonVoiceDataModule
from bolts.simclr import SpeechSimClr
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin, DDPShardedPlugin

simclr_datamodule = UnsupervisedCommonVoiceDataModule()
simclr_datamodule.prepare_data()
simclr_datamodule.setup(stage='test')

simclr = SpeechSimClr.load_from_checkpoint('training_artifacts/weights_best/epoch=35-step=271331.ckpt', num_samples=simclr_datamodule.num_test_samples(), strict=True)

trainer = pl.Trainer(
    default_root_dir=config.trainer.default_root_dir,
    gpus=config.trainer.num_gpus,
    accelerator='ddp' if config.trainer.num_gpus > 1 else None,
    plugins=DDPPlugin(
        find_unused_parameters=False) if config.trainer.num_gpus > 1 else None,
    num_nodes=config.trainer.num_nodes,
    log_every_n_steps=config.trainer.log_every_n_steps,
    precision=config.trainer.precision
)

trainer.test(simclr, datamodule=simclr_datamodule)
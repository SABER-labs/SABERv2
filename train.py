import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.config import config
from bolts.unsupervised_data import UnsupervisedCommonVoiceDataModule
from bolts.simclr import SpeechSimClr
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin, DDPShardedPlugin

ssl_datamodule = UnsupervisedCommonVoiceDataModule()
ssl_datamodule.prepare_data()
ssl_datamodule.setup(stage='fit')

ssl = SpeechSimClr(num_samples=ssl_datamodule.num_train_samples())
# ssl = BarlowTwins(num_samples=ssl_datamodule.num_train_samples())
# ssl = Simsiam(num_samples=ssl_datamodule.num_train_samples())
# ssl = AggregatedEntropicTwins(num_samples=ssl_datamodule.num_train_samples())

logger = TensorBoardLogger(
    os.path.join(
        config.trainer.default_root_dir,
        config.trainer.tensorboard_logdir),
    name='ssl')

model_checkpoint = ModelCheckpoint(
    dirpath=os.path.join(config.trainer.default_root_dir,
                         config.trainer.savewieghts_dir),
    save_last=True,
    save_top_k=1,
    monitor='train_loss'
)
lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks = [model_checkpoint, lr_monitor]

trainer = pl.Trainer(
    default_root_dir=config.trainer.default_root_dir,
    gpus=config.trainer.num_gpus,
    max_epochs=config.trainer.max_epochs,
    accelerator='ddp_sharded' if config.trainer.num_gpus > 1 else None,
    plugins=DDPShardedPlugin(find_unused_parameters=False, sync_batchnorm=True) if config.trainer.num_gpus > 1 else None,
    num_nodes=config.trainer.num_nodes,
    log_every_n_steps=config.trainer.log_every_n_steps,
    gradient_clip_val=config.trainer.gradient_clip_val,
    precision=config.trainer.precision,
    callbacks=callbacks,
    fast_dev_run=config.trainer.fast_dev_run,
    logger=logger,
    terminate_on_nan=True,
    sync_batchnorm=True if config.trainer.num_gpus > 1 else False,
    # overfit_batches=0.05,
    # track_grad_norm=2,
    # overfit_batches=0.01,
    # profiler=True
    # weights_summary='full'
    # resume_from_checkpoint=os.path.join(config.trainer.default_root_dir, config.trainer.savewieghts_dir)
)

trainer.fit(ssl, datamodule=ssl_datamodule)

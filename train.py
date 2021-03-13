import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.config import config
from bolts.unsupervised_data import UnsupervisedCommonVoiceDataModule
from bolts.simclr import SpeechSimClr
from pytorch_lightning.loggers import TensorBoardLogger

simclr_datamodule = UnsupervisedCommonVoiceDataModule()
simclr_datamodule.prepare_data()
simclr_datamodule.setup(stage='fit')

simclr = SpeechSimClr(num_train_samples=simclr_datamodule.num_train_samples())
logger = TensorBoardLogger(os.path.join(
	config.trainer.default_root_dir, config.trainer.tensorboard_logdir), name='simclr')

model_checkpoint = ModelCheckpoint(
	dirpath=os.path.join(config.trainer.default_root_dir, config.trainer.savewieghts_dir), 
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
	accelerator='ddp' if config.trainer.num_gpus > 1 else None,
	plugins='ddp_sharded' if config.trainer.num_gpus > 1 else None,
	num_nodes=config.trainer.num_nodes,
	log_every_n_steps=config.trainer.log_every_n_steps,
	gradient_clip_val=config.trainer.gradient_clip_val,
	precision=config.trainer.precision,
	callbacks=callbacks,
	fast_dev_run=config.trainer.fast_dev_run,
	logger=logger,
	terminate_on_nan=True,
	# resume_from_checkpoint=os.path.join(config.trainer.default_root_dir, config.trainer.savewieghts_dir)
)

trainer.fit(simclr, datamodule=simclr_datamodule)

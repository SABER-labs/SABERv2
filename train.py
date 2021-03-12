import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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
	monitor='simclr_loss'
)

callbacks = [model_checkpoint]

trainer = pl.Trainer(
	default_root_dir=config.trainer.default_root_dir,
	gpus=config.trainer.num_gpus,
	max_epochs=config.trainer.max_epochs,
	distributed_backend='ddp2' if config.trainer.num_gpus > 1 else None,
	sync_batchnorm=True if config.trainer.num_gpus > 1 else False,
	num_nodes=config.trainer.num_nodes,
	log_every_n_steps=config.trainer.log_every_n_steps,
	gradient_clip_val=config.trainer.gradient_clip_val,
	precision=config.trainer.precision,
	callbacks=callbacks,
	fast_dev_run=config.trainer.fast_dev_run,
	logger=logger
)

trainer.fit(simclr, datamodule=simclr_datamodule)

![alt text](icons/character+fat+game+hero+inkcontober+movie+icon-1320183878106104615_24.png) SABER - Semi-Supervised Audio Baseline for Easy Reproduction
=====
Easily reproducible baselines for automatic speech recognition using semi-supervised contrastive learning.

## Data Preparation
* Download [CommonVoice English Dataset](https://commonvoice.mozilla.org/en/datasets)
* Setup `config.toml` to use the paths where data was downloaded.
* Install requirements using `pip3 install -r requirements.txt`
* Prepare data using `python3 -m dataset.prepare`

## Train
* Train using `python3 -m train`

## Logging
* Start tensorboard using `tensorboard --logdir training_artifacts/tb_logs`

## TODOS
* supervised training and dataset
* Check online evaluator piece from [Pybolts Simclr](https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py)
* ~~Add more logs.~~
* ~~streaming convnets model~~
* ~~save and load projection weighs for training~~
* ~~Check if anything is missing from [Athena Simclr](https://github.com/athena-team/athena/blob/simclr/athena/models/simclr.py)~~
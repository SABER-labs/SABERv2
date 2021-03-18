import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.config import config
from typing import Optional
from transforms.audio import RandomSoxAugmentations, NoSoxAugmentations
from transforms.mfsc import ToMelSpec, SpecAug
from dataset.test_dataset import SimClrTestDataset
import time
from pytorch_lightning.utilities import move_data_to_device

class UnsupervisedCommonVoiceDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.sox_augmentations = RandomSoxAugmentations(config.dataset.sample_rate)
        self.no_augmentation = NoSoxAugmentations(config.dataset.sample_rate)
        self.mel_then_specaug = torch.jit.script(torch.nn.Sequential(ToMelSpec(), SpecAug()))
        self.only_mel = torch.jit.script(torch.nn.Sequential(ToMelSpec()))

    def setup(self, stage: Optional[str] = None):
        if stage == 'test':
            self.test_dataset = SimClrTestDataset(
                root=config.dataset.test_root, tsv=config.dataset.test)
            self.transform = self.only_mel
            self.augmentation = self.no_augmentation
        else:
            self.unsupervised_dataset = torchaudio.datasets.COMMONVOICE(
                root=config.dataset.root, tsv=config.dataset.unsupervised_train)
            self.transform = self.mel_then_specaug
            self.augmentation = self.sox_augmentations


    def num_train_samples(self):
        return len(self.unsupervised_dataset)

    def num_test_samples(self):
        return len(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.unsupervised_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=len(self.test_dataset),
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn
        )

    def transfer_batch_to_device(self, batch, device):
        device = device or self.device
        self.transform = self.transform.to(device)
        return move_data_to_device(batch, device)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        input_a, input_b, input_a_lengths, input_b_lengths = batch
        input_a = self.transform(input_a)
        input_b = self.transform(input_b)
        input_a_lengths = (input_a_lengths / (config.audio.model_sample_rate/1000 * config.audio.stride_in_ms)).ceil_()
        input_b_lengths = (input_b_lengths / (config.audio.model_sample_rate/1000 * config.audio.stride_in_ms)).ceil_()
        return (input_a, input_b, input_a_lengths, input_b_lengths)

    # input: batch -> (waveform, sample_rate, dictionary)
    # returns: (aug1, aug2, aug1_len, aug2_len) where aug1 == (batch, time)
    def _collate_fn(self, batch):

        raw_inputs = [b[0] for b in batch if b]
        input_a = [self.augmentation(raw_input).transpose(1, 0) for raw_input in raw_inputs]
        input_b = [self.augmentation(raw_input).transpose(1, 0) for raw_input in raw_inputs]

        input_a_lengths = torch.tensor(
            [t.size(0) for t in input_a],
            dtype=torch.int32,
            device=input_a[0].device,
        )
        input_b_lengths = torch.tensor(
            [t.size(0) for t in input_b],
            dtype=torch.int32,
            device=input_b[0].device,
        )

        input_a = torch.nn.utils.rnn.pad_sequence(input_a, batch_first=True).transpose(1, -1).squeeze(1)
        input_b = torch.nn.utils.rnn.pad_sequence(input_b, batch_first=True).transpose(1, -1).squeeze(1)

        return (input_a, input_b, input_a_lengths, input_b_lengths)

if __name__ == "__main__":
    loader = UnsupervisedCommonVoiceDataModule()
    loader.setup()
    for i, batch in enumerate(loader.train_dataloader()):
        print(batch[0].shape, batch[1].shape, batch[2], batch[3])
        if i > 0 and i % 20 == 0:
            break
        
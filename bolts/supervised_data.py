import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from utils.config import config
from typing import Optional
from transforms.audio import RandomSoxAugmentations, NoSoxAugmentations
from transforms.mfsc import ToMelSpec, SpecAug
from dataset.test_dataset import SimClrTestDataset
import time
from pytorch_lightning.utilities import move_data_to_device
import sentencepiece as spm
import os

class SupervisedCommonVoiceDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.sox_augmentations = RandomSoxAugmentations(config.dataset.sample_rate)
        self.no_augmentation = NoSoxAugmentations(config.dataset.sample_rate)
        self.mel_then_specaug = torch.jit.script(torch.nn.Sequential(ToMelSpec(), SpecAug()))
        self.only_mel = torch.jit.script(torch.nn.Sequential(ToMelSpec()))
        self.pad = 0
        self.unk = 1
        self.eos = 2
        self.bos = 3

    def prepare_data(self):
        train_cmd = '--input={input} --model_prefix={prefix} --vocab_size={sz}\
         --max_sentencepiece_length=3 --character_coverage=1.0 --model_type=unigram\
          --hard_vocab_limit=false --split_by_unicode_script=false\
           --pad_id={pad} --unk_id={unk} --bos_id={bos} --eos_id={eos} \
           --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS]'.format(
            input=config.supervised_train.corpus, prefix=config.supervised_train.language_modelpath,\
             sz=config.supervised_train.nclasses, pad=self.pad, unk=self.unk, bos=self.bos, eos=self.eos
        )

        spm.SentencePieceTrainer.Train(train_cmd)
        
        # print ("file_name", f)
        
    def setup(self, stage: Optional[str] = None):

            self.supervised = torchaudio.datasets.COMMONVOICE(
                root=config.dataset.root, tsv=config.dataset.supervised_train)
            self.transform = self.mel_then_specaug
            self.augmentation = self.sox_augmentations
            size = self.supervised.__len__()
            self.train, self.val = random_split(self.supervised, [int(0.9 * size), size - int(0.9*size)])
            f = config.supervised_train.language_modelpath+".model"
            self.tokenizer = spm.SentencePieceProcessor(model_file=f)
            


    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
            drop_last=True,
            # shuffle=True,
            collate_fn=self._collate_fn
        )

    def transfer_batch_to_device(self, batch, device):
        device = device or self.device
        self.transform = self.transform.to(device)
        return move_data_to_device(batch, device)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        input_a, input_a_lengths, target, target_lengths = batch
        input_a = self.transform(input_a)
        input_a_lengths = (input_a_lengths / (config.audio.model_sample_rate/1000 * 2*config.audio.stride_in_ms)).ceil_().int()
        return (input_a, input_a_lengths, target, target_lengths)

    def num_train_samples(self):
        return len(self.supervised)
    
    def _collate_fn(self, batch):
        raw_inputs = [b[0] for b in batch if b]
        input_a = [self.augmentation(raw_input).transpose(1, 0) for raw_input in raw_inputs]
        target_sentences = [torch.tensor(self.tokenizer.encode(b[2]['sentence']), \
        dtype=torch.int32, device=input_a[0].device) for b in batch if b]
        input_a_lengths = torch.tensor(
            [x.shape[0] for x in input_a],
            dtype=torch.int32,
            device=input_a[0].device,
        )
        input_a = torch.nn.utils.rnn.pad_sequence(input_a, batch_first=True).transpose(1, -1).squeeze(1)
        target_lengths = torch.tensor([len(x) for x in target_sentences], dtype=torch.int32, device=input_a[0].device)
        target_sentences = torch.nn.utils.rnn.pad_sequence(target_sentences, True, self.pad)
        return (input_a, input_a_lengths, target_sentences, target_lengths)

if __name__ == "__main__":
    loader = SupervisedCommonVoiceDataModule()
    loader.setup()
    for i, batch in enumerate(loader.train_dataloader()):
        print(batch[0].shape, batch[1].shape, batch[2], batch[3])
        if i > 0 and i % 20 == 0:
            break

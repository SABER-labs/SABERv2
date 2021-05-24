import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.config import config
from typing import Optional
from transforms.audio import RandomSoxAugmentations, NoSoxAugmentations
from transforms.mfsc import ToMelSpec, SpecAug
from pytorch_lightning.utilities import move_data_to_device
import sentencepiece as spm
import os
import pandas as pd
import shutil
import jiwer


def cleaner_func():
    transformation = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveEmptyStrings()
    ])

    def cleanup(text):
        return transformation(text)

    return cleanup


def build_text_corpus():
    root = config.dataset.root
    all_files = [os.path.join(root, file) for file in [
        config.dataset.train, config.dataset.test]]
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, sep='\t', index_col=None, header=0)
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    a_list = frame['sentence'].values
    cleaner = cleaner_func()
    with open(os.path.join(config.dataset.root, config.dataset.text_corpus), 'w') as textfile:
        for element in a_list:
            textfile.write(cleaner(str(element)) + "\n")


class SupervisedCommonVoiceDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        self.sox_augmentations = RandomSoxAugmentations(
            config.dataset.sample_rate)
        self.no_augmentation = NoSoxAugmentations(config.dataset.sample_rate)
        self.mel_then_specaug = torch.jit.script(
            torch.nn.Sequential(ToMelSpec(), SpecAug()))
        self.only_mel = torch.jit.script(torch.nn.Sequential(ToMelSpec()))
        self.cleaner = cleaner_func()

    def prepare_data(self):

        if not os.path.exists(os.path.join(config.dataset.root, f'{config.dataset.spe_prefix}.vocab')):

            build_text_corpus()

            text_corpus_path = os.path.join(
                config.dataset.root, config.dataset.text_corpus)

            train_cmd = '--input={input} --model_prefix={prefix} --vocab_size={sz} \
            --max_sentencepiece_length=2 --character_coverage=1.0 --model_type=unigram \
            --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1'.format(
                input=text_corpus_path, prefix=config.dataset.spe_prefix,
                sz=config.dataset.n_classes - 1)

            spm.SentencePieceTrainer.Train(train_cmd)

            shutil.move(f'{config.dataset.spe_prefix}.vocab',
                        config.dataset.root)
            shutil.move(f'{config.dataset.spe_prefix}.model',
                        config.dataset.root)

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = torchaudio.datasets.COMMONVOICE(
            root=config.dataset.root, tsv=config.dataset.supervised_train)
        self.val_dataset = torchaudio.datasets.COMMONVOICE(
            root=config.dataset.root, tsv=config.dataset.test)

        self.set_stage(stage)

        self.tokenizer = spm.SentencePieceProcessor(
            model_file=f'{os.path.join(config.dataset.root, config.dataset.spe_prefix)}.model')

    def get_tokenizer(self):
        return self.tokenizer

    def clean_text(self, text):
        return self.cleaner(text)

    def decode_model_output(self, targets):
        new_targets = []
        for target in targets:
            new_target = []
            for i, token in enumerate(target):
                if ((i == 0) or (i > 0 and token != target[i-1])) and (token < config.dataset.n_classes - 1):
                    new_target.append(token)
            new_targets.append(new_target)
        return new_targets

    def set_stage(self, stage):
        if stage in ["val", "test"]:
            self.transform = self.only_mel
            self.augmentation = self.no_augmentation
        else:
            self.transform = self.mel_then_specaug
            self.augmentation = self.sox_augmentations

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=self._collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=config.dataloader.batch_size,
            num_workers=config.dataloader.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )

    def transfer_batch_to_device(self, batch, device):
        device = device or self.device
        self.transform = self.transform.to(device)
        return move_data_to_device(batch, device)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        input_a, input_a_lengths, target, target_lengths = batch
        input_a = self.transform(input_a)
        input_a_lengths = (input_a_lengths / (config.audio.model_sample_rate /
                           1000 * config.audio.stride_in_ms)).ceil_().int()
        return (input_a, input_a_lengths, target, target_lengths)

    def num_train_samples(self):
        return len(self.train_dataset)

    def num_test_samples(self):
        return len(self.test_dataset)

    def _collate_fn(self, batch):
        raw_inputs = [b[0] for b in batch if b]
        input_a = [self.augmentation(raw_input).transpose(1, 0)
                   for raw_input in raw_inputs]
        target_sentences = [torch.tensor(self.tokenizer.encode(self.clean_text(b[2]['sentence']), enable_sampling=True, alpha=0.9, nbest_size=10),
                                         dtype=torch.int32, device=input_a[0].device) for b in batch if b]

        input_a_lengths = torch.tensor(
            [x.shape[0] for x in input_a],
            dtype=torch.int32,
            device=input_a[0].device,
        )
        input_a = torch.nn.utils.rnn.pad_sequence(
            input_a, batch_first=True).transpose(1, -1).squeeze(1)
        target_lengths = torch.tensor(
            [x.size(0) for x in target_sentences], dtype=torch.int32, device=input_a[0].device)
        target_sentences = torch.nn.utils.rnn.pad_sequence(
            target_sentences, batch_first=True)
        return (input_a, input_a_lengths, target_sentences, target_lengths)


if __name__ == "__main__":
    from tqdm import tqdm
    loader = SupervisedCommonVoiceDataModule()
    loader.prepare_data()
    loader.setup('train')
    for i, batch in enumerate(tqdm(loader.train_dataloader())):
        batch = loader.transfer_batch_to_device(batch, 'cuda')
        batch = loader.on_after_batch_transfer(batch, 0)
        ref = loader.get_tokenizer().decode(batch[2].detach().cpu().numpy().tolist())
        encoded_ref_lengths = batch[3].detach().cpu().numpy().tolist()
        lengths = [len(rf) for rf in ref]
        input_lengths = batch[1].detach().cpu().numpy().tolist()
        input_lengths = [input_length // 4 for  input_length in input_lengths]

        for i, (in_len, tar_len, sen_len) in enumerate(zip(input_lengths, encoded_ref_lengths, lengths)):
            if in_len <= tar_len:
                print(f"Sentence with issue was: {ref[i]}, input_length: {in_len}, target_length: {tar_len}, sentence_length: {sen_len}")

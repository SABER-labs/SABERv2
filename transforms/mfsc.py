import torch
import torch.nn as nn
import torchaudio
from utils.config import config
import os


class ToMelSpec(torch.nn.Module):

    def __init__(
            self,
            input_sample_rate: int = config.audio.model_sample_rate):
        super().__init__()
        self.to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=input_sample_rate,
            win_length=int(
                input_sample_rate /
                1000 *
                config.audio.window_size_in_ms),
            hop_length=int(
                input_sample_rate /
                1000 *
                config.audio.stride_in_ms),
            n_fft=int(
                input_sample_rate /
                1000 *
                config.audio.window_size_in_ms),
            f_max=input_sample_rate /
            2,
            n_mels=config.audio.n_mels)
    '''
        input  -> (.., time)
        output -> (.., n_mels, time)
    '''

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.to_melspec(sample)


class SpecAug(torch.nn.Module):

    def __init__(
            self,
            input_sample_rate: int = config.audio.model_sample_rate):
        super().__init__()
        self.spec_aug = nn.Sequential(
            *
            ([torchaudio.transforms.FrequencyMasking(
                freq_mask_param=config.spec_aug.freq_len, iid_masks=True)] *
             config.spec_aug.freq_n),
            *
            ([torchaudio.transforms.TimeMasking(
                time_mask_param=config.spec_aug.time_len, iid_masks=True)] *
             config.spec_aug.time_n),)
    '''
        input  -> (.., n_mels, time)
        output -> (.., n_mels, time)
    '''

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.spec_aug(sample)


if __name__ == "__main__":
    import time
    import math
    # waveform, samplerate = torchaudio.load(os.path.join(config.dataset.root, "clips", "common_voice_en_572372.mp3"))
    waveform, samplerate = torchaudio.load(os.path.join(
        config.dataset.root, "clips", "common_voice_en_17970627.mp3"))
    augmention = torch.jit.script(torch.nn.Sequential(
        ToMelSpec(samplerate), SpecAug(samplerate))).cuda()
    for i in range(100):
        start = time.process_time()
        augmented_waveform = augmention(waveform.cuda()).squeeze(0)
        calculated_shape = math.ceil(waveform.size(
            1) / (samplerate / 1000 * config.audio.stride_in_ms))
        print(
            f"time taken for augmentation: {(time.process_time() - start) * 1000}ms, \
            augmented shape: {augmented_waveform.shape}, calculated shape: {calculated_shape}")

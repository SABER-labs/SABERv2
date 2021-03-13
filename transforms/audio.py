import torch
import torch.nn as nn
import torchaudio
from utils.config import config
from numpy.random import uniform
from torchaudio.sox_effects import apply_effects_tensor
import os
import random
import math

'''
List of transforms available in torchaudio-sox:
allpass, band, bandpass, bandreject, bass, bend, biquad, chorus,
compand, contrast, dcshift, deemph, delay, dither, divide,
earwax, echo, echos, equalizer, fade, fir, firfit,
gain, highpass, hilbert, loudness, lowpass, mcompand, norm,
overdrive, pad, phaser, pitch, rate, remix, repeat,
reverse, riaa, silence, sinc, speed, stat, stats,
swap, synth, tempo, treble, tremolo, trim, upsample,
vol
'''

class ToMulaw(torch.nn.Module):

    def __init__(self, input_sample_rate: int=16000):
        super().__init__()
        transform_list = [torchaudio.transforms.MuLawEncoding(), torchaudio.transforms.MuLawDecoding()]
        if input_sample_rate != 8000:
            transform_list = [torchaudio.transforms.Resample(orig_freq=input_sample_rate, new_freq=8000)] + transform_list
        transforms= nn.Sequential(*transform_list)
        self.to_mulaw = torch.jit.script(transforms)

    def forward(self, sample: torch.Tensor):
        return self.to_mulaw(sample)

class RandomSoxAugmentations(object):

    def __init__(self, input_sample_rate):
        self.sample_rate = input_sample_rate
        self.augmentations = [
            ['vol', str(uniform(*config.augmentations.vol_range_in_db)), 'dB'],
            ['pitch', '-q', str(uniform(*config.augmentations.pitch_range_in_cents))],
            ['tempo', '-q', '-s', str(uniform(*config.augmentations.tempo_range))],
            ['lowpass', '-2', str(config.augmentations.lowpass_cutoff)]
        ]
        self.augmentation_prob = config.augmentations.apply_prob
        self.noise_file_list = [
            os.path.join(config.noises.noises_root, path.strip()) 
            for path in 
            open(os.path.join(config.noises.noises_root, config.noises.noisefilelist), 'r').read().split("\n") if path
            ]

    def __add_noise(self, sample: torch.Tensor):
        noise, _ = torchaudio.load(filepath = random.choice(self.noise_file_list))
        if noise.size(1) >= sample.size(1):
            noise = noise[:, :sample.size(1)]
        else:
            times_to_tile = int(sample.size(1) / noise.size(1))
            noise = torch.cat((noise.tile((1, times_to_tile)), noise[:, :(sample.size(1) - noise.size(1) * times_to_tile)]), dim=1)
        speech_power = sample.norm(p=2)
        noise_power = noise.norm(p=2)
        snr = math.exp(uniform(*config.noises.snr_range) / 10)
        scale = snr * noise_power / speech_power
        noisy_speech = (scale * sample + noise) / 2
        return noisy_speech

    def __call__(self, sample: torch.Tensor):
        effects = [augmentation for augmentation in self.augmentations if uniform(0.0, 1.0) <= self.augmentation_prob] + [['rate', str(config.audio.model_sample_rate)]]
        augmented_sample, _ = apply_effects_tensor(sample, self.sample_rate, effects, channels_first=True)
        if uniform(0.0, 1.0) <= self.augmentation_prob:
            augmented_sample = self.__add_noise(augmented_sample)
        return augmented_sample

class NoSoxAugmentations(object):

    def __init__(self, input_sample_rate):
        self.sample_rate = input_sample_rate
        self.effect = [
            ['rate', str(config.audio.model_sample_rate)]
        ]

    def __call__(self, sample: torch.Tensor):
        augmented_sample, _ = apply_effects_tensor(sample, self.sample_rate, self.effect, channels_first=True)
        return augmented_sample

if __name__ == "__main__":
    import time
    waveform, samplerate = torchaudio.load(os.path.join(config.dataset.root, "clips", "common_voice_en_572372.mp3"))
    augmention = RandomSoxAugmentations(samplerate)
    for i in range(100):
        start = time.process_time()
        augmented_waveform = augmention(waveform)
        print(f"time taken for augmentation: {(time.process_time() - start) * 1000}ms, augmented shape: {augmented_waveform.shape}")
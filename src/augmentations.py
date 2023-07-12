from typing import Union, Tuple, List

import torch
import torch.nn as nn

import torchaudio.functional as F
from torchaudio.transforms import FrequencyMasking, TimeMasking, TimeStretch


def batchwise_gain(wav: torch.Tensor, gain_db: Union[int, float, torch.Tensor]) -> torch.Tensor:
    """
    Modified version of torchaudio.functional.gain to apply to whole batch of waveforms with randomly sampled gains
    """
    ratio = 10 ** (gain_db / 20)
    return torch.einsum('...,...ct->...ct', ratio, wav)


class RandomWhiteNoise(nn.Module):
    """
    Callable class to apply white noise to audio according to specified signal-to-noise ratio (SNR, in dB).
    :param snr: scalar value in representing the signal-to-noise ratio in dB
                a value of 0 SNR represents noise equally loud as the main signal while
                a value of 10 SNR represents noise which is 10 dB quieter than the main signal.
                negative values represent situations where noise is louder than the original signal.
    """
    def __init__(self, min_snr: int, max_snr: int):
        super().__init__()
        assert isinstance(min_snr, int) or isinstance(min_snr, float), "Min. gain must be a scalar numeric value"
        assert isinstance(max_snr, int) or isinstance(max_snr, float), "Max. gain must be a scalar numeric value"
        assert (delta := max_snr - min_snr) > 0, "Max. gain must be greater than min. gain"

        self.device = "cpu"
        self.delta = delta
        self.min_snr = min_snr
        self.max_snr = max_snr

    def to(self, device):
        self.device = device
        return self

    def sample_snr(self, wav: torch.Tensor) -> torch.Tensor:
        assert wav.dim() == 3, "Expected 3D tensor (batch, channel, time)"
        batch_size, n_channels = wav.size()[:-1]
        snr_t = torch.rand(size=(batch_size,), device=self.device) * self.delta + self.min_snr
        return snr_t.expand(n_channels, -1).transpose(-2, -1)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(wav, device=self.device)
        snr_t = self.sample_snr(wav)
        return F.add_noise(wav, noise, snr=snr_t)


class RandomGain(nn.Module):
    """
    Callable class to apply randomly sampled gain to batches of waveforms
    :param min_gain:
    :param max_gain:
    :param mode: 'per_sample', 'per_batch', 'per_channel'
    :param distrib: 'uniform', 'normal'
    """
    def __init__(self, min_gain: Union[int, float], max_gain: Union[int, float]):
        super().__init__()
        assert isinstance(min_gain, int) or isinstance(min_gain, float), "Min. gain must be a scalar numeric value"
        assert isinstance(max_gain, int) or isinstance(max_gain, float), "Max. gain must be a scalar numeric value"
        assert (delta := max_gain - min_gain) > 0, "Max. gain must be greater than min. gain"

        self.device = "cpu"
        self.delta = delta
        self.min_gain = min_gain
        self.max_gain = max_gain

    def to(self, device):
        self.device = device
        return self

    def sample_gain(self, wav: torch.Tensor) -> torch.Tensor:
        assert wav.dim() == 3, "Expected 3D tensor (batch, channel, time)"
        batch_size = wav.size(-3)
        return torch.rand(size=(batch_size,), device=self.device) * self.delta + self.min_gain

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Applies sample-wise random gain to the input batch of waveform samples
        :param wav: tensor of size ([batch], channels, time,)
        :return: same batch of waveforms with randomly applied volume adjustment (gain)
        """
        gain_db = self.sample_gain(wav)
        return batchwise_gain(wav, gain_db=gain_db)


class RandomPitchShift(nn.Module):
    # Pitch alteration would be a desirable way to augment our song extracts, but
    # it induces quite a high computational overhead, so it is not implemented
    pass


class WaveformAugment(nn.Module):
    """
    Applies random gain and random white noise to batches of waveforms
    """
    def __init__(
            self,
            white_noise_range: Union[List[Union[int, float]], Tuple[Union[int, float]]],
            gain_db_range: Union[List[Union[int, float]], Tuple[Union[int, float]]],
    ):
        super().__init__()
        assert not (white_noise_range is None and gain_db_range is None), "No transform to apply: both ranges are None"
        self.white_noise = RandomWhiteNoise(*white_noise_range) if white_noise_range is not None else lambda x: x
        self.random_gain = RandomGain(*gain_db_range) if gain_db_range is not None else lambda x: x

    def to(self, device):
        if isinstance(self.white_noise, nn.Module):
            self.white_noise = self.white_noise.to(device)
        if isinstance(self.random_gain, nn.Module):
            self.random_gain = self.random_gain.to(device)
        return self

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        wav = self.random_gain(wav)
        wav = self.white_noise(wav)
        return wav


class SpecAugment(nn.Module):
    """
    Callable module to apply random spectral and temporal masking on spectrogram batches as a mean of data augmentation
    """
    def __init__(
            self,
            max_freq_mask_len: int,
            max_time_mask_len: int,
            max_time_mask_frac: float = 0.3,
    ):
        super().__init__()
        assert not (max_freq_mask_len is None and max_time_mask_len is None), "No transform to apply: both are None"

        self.freq_mask = FrequencyMasking(
            freq_mask_param=max_freq_mask_len,
            iid_masks=True) if max_freq_mask_len is not None else lambda x: x
        self.time_mask = TimeMasking(
            time_mask_param=max_time_mask_len,
            iid_masks=True,
            p=max_time_mask_frac) if max_time_mask_len is not None else lambda x: x

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Applies spectral and/or temporal masking to the input batch of spectrograms
        :param spec: spectrogram batches of size (batch, channel, freq, time)
        :return: the batch of spectrograms with spectral and/or temporal masking applied (of same size)
        """
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec


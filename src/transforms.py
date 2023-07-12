import torch
import torch.nn as nn
import torchaudio.transforms as tt


class Log(nn.Module):
    def __init__(self, p=None, eps=1e-8):
        super().__init__()
        if p is not None:
            assert isinstance(p, float) or isinstance(p, int) and p > 1
            self.log_fn = lambda x: torch.log(x + eps) / torch.log(p)
        else:
            self.log_fn = lambda x: torch.log(x + eps)

    def forward(self, x):
        return self.log_fn(x)


class Log1p(nn.Module):
    def __init__(self, p=None):
        super().__init__()
        if p is not None:
            assert isinstance(p, float) or isinstance(p, int) and p > 1
            self.log_fn = lambda x: torch.log1p(x) / torch.log(p)
        else:
            self.log_fn = lambda x: torch.log1p(x)

    def forward(self, x):
        return self.log_fn(x)


class StdMeanScaling(nn.Module):
    def __init__(self, axis=None):
        super().__init__()
        self.compute_std_mean = lambda x: torch.std_mean(x, dim=axis, keepdim=True)

    def forward(self, x):
        std, mean = self.compute_std_mean(x)
        return (x - mean) / std


def rawspec():
    return tt.Spectrogram(n_fft=1024, win_length=1024, hop_length=512, power=1.0)


def powerspec(log=False, normalize=False, norm_axis=None):
    transform = nn.Sequential()
    if log:
        transform.add_module("Log", Log1p())
    if normalize:
        transform.add_module("Std-Mean-Scaling", StdMeanScaling(axis=norm_axis))
    transform.add_module("Power-Spectrogram", tt.Spectrogram(n_fft=1024, win_length=1024, hop_length=512, power=2.0))
    return transform


def logspec(normalize=None, norm_axis=None):
    return nn.Sequential(powerspec(normalize, norm_axis), Log())


def melspec():
    raise NotImplementedError


def mfcc():
    raise NotImplementedError


def lfcc():
    raise NotImplementedError


def imfcc():
    raise NotImplementedError

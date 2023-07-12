import json
import os.path

import torch

from . import transforms
from .models import cnn


def get_transform(feature_name, **feature_kwargs):
    """
    Given a transform name, retrieves it from defined transforms and instantiates it with given keyword arguments
    :param feature_name: name of the transform, e.g. 'Spectrogram', 'MelSpectrogram', 'MFCC' ...
    :param feature_kwargs: keyword arguments to feed the transform module, e.g. 'n_fft', 'n_filters' ...
    :return: instantiated transform function as a nn.Module object
    """
    try:
        transform = getattr(transforms, feature_name)
    except AttributeError as err:
        print(f"Feature '{feature_name}' could not be found.")
        raise err.with_traceback(err.__traceback__)

    return transform(**feature_kwargs)


def get_model(model_name, *model_args, **model_kwargs):
    """
    Given a model name, retrieves it from defined models and instantiates it with given arguments
    :param model_name: name of the model to use, e.g. 'CNN', 'ResNet' ...
    :param model_args: non-keyword arguments to feed the model constructor
    :param model_kwargs: keyword argument to feed the model constructor
    :return: instantiated and initialized torch model as nn.Module object
    """
    found = False
    for module in [cnn]:
        try:
            model = getattr(module, model_name)
            found = True
            break
        except AttributeError:
            continue
    if not found:
        raise AttributeError(f"Model '{model_name}' could not be found.")

    return model(*model_args, **model_kwargs)


def parse_kwargs_arguments(argument: str):
    """
    Returns a set of keyword arguments as a dictionary by reading a specified JSON file or parsing a JSON-like string
    :param argument: path to JSON file or JSON-like string containing keyword arguments
    :return: dictionary containing keyword arguments
    """
    if argument is None:
        kwargs = dict()
    elif os.path.isfile(argument):
        kwargs = json.load(open(argument))
    else:
        kwargs = json.loads(argument)
    return kwargs


def slice_audio(
        wav: torch.Tensor,
        slice_duration: float = 4.0,
        overlap: float = 0.75,
        sample_rate: int = 22_050,
        max_duration=60
):
    assert 0.0 <= overlap < 1.0
    offset = 0
    slice_frames = round(slice_duration * sample_rate)
    hop_frames = round(slice_frames * (1 - overlap))
    num_frames = wav.size(-1)
    delta = num_frames - (offset + slice_frames)

    slices = list()
    while delta > 0:
        delta = num_frames - (offset + slice_frames)
        if delta < 0:
            slices.append(wav[offset:].repeat(2)[:slice_frames])
        else:
            slices.append(wav[offset: offset+slice_frames])
        offset += hop_frames

    return slices

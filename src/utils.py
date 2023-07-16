import json
import os.path
from io import BytesIO
from typing import Any

import ffmpeg
import streamlit as st

import torch
from pytube import YouTube

import plotly.express as px

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
        max_duration=600,
):
    assert 0.0 <= overlap < 1.0
    offset = 0
    slice_frames = round(slice_duration * sample_rate)
    hop_frames = round(slice_frames * (1 - overlap))
    num_frames = wav.size(-1)
    delta = num_frames - (offset + slice_frames)

    max_offset = round(sample_rate * max_duration)

    slices = list()
    while delta > 0 and offset < max_offset:
        delta = num_frames - (offset + slice_frames)
        if delta < 0:
            slices.append(wav[...,offset:].repeat(1, 2)[...,:slice_frames])
        else:
            slices.append(wav[...,offset: offset+slice_frames])
        offset += hop_frames

    return slices


@st.cache_resource
def load_model_and_transform(exp_dir: str, checkpoint: str = "accuracy", model_type: str = 'CNN'):
    with open(os.path.join(exp_dir, "config.json"), 'r') as f:
        config = json.load(f)
    model_kwargs = parse_kwargs_arguments(config["model_kwargs"])
    model: torch.nn.Module = get_model(model_name=model_type, num_classes=10, **model_kwargs)

    feat_name = config["feature"]
    feat_kwargs = parse_kwargs_arguments(config["feature_kwargs"])
    transform = get_transform(feat_name, **feat_kwargs)

    if checkpoint == "accuracy":
        cp = os.path.join(exp_dir, "checkpoints", "best_acc_1.pt")
    elif checkpoint == "loss":
        cp = os.path.join(exp_dir, "checkpoints", "best_loss.pt")
    else:
        cp = os.path.join(exp_dir, "checkpoints", f"{checkpoint}.pt")

    state_dict = torch.load(cp, map_location="cpu")
    model.load_state_dict(state_dict)

    return model, transform


def get_activation(name, out):
    def hook(i, o, output):
        out[name] = output.detach()
    return hook


def ffmpeg_reencode(f, target_ar: str = '22050', duration=None):
    input_options = {"ac": 1}
    if duration is not None:
        input_options["t"] = duration
    audio_stream, err = (
        ffmpeg
        .input(f, **input_options)
        .output("pipe:", format='wav',
                acodec='pcm_s16le',
                ar=target_ar,
                ac=1)  # Select WAV output format, and pcm_s16le auidio codec.
        .run(capture_stdout=True)
    )
    audio_io = BytesIO(audio_stream)
    return audio_io, err


def wav_tensor_from_stream(s):
    wav = torch.frombuffer(s, dtype=torch.int16).to(torch.float32)
    if wav.dim() == 1:
        wav = wav.unqueeze(0)
    wav = wav[:1, :]
    wav -= torch.mean(wav)
    wav /= torch.max(torch.abs(wav))
    return wav


def get_audio_stream_from_youtube(url: str, target_ar: str = '22050', duration=None):
    yt = YouTube(url)

    stream_url = yt.streams.filter(only_audio=True).first().url  # Get the URL of the video stream

    # Probe the audio streams (use it in case you need information like sample rate):
    # probe = ffmpeg.probe(stream_url)
    # audio_streams = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    # sr = audio_streams['sample_rate']

    # Read audio into memory buffer.
    # Get the audio using stdout pipe of ffmpeg sub-process.
    # The audio is transcoded to PCM codec in WAC container.
    audio_io, err = ffmpeg_reencode(stream_url, target_ar, duration)

    return audio_io, err


def ffmpeg_reencode_from_uploaded_file(file, target_ar: str = '22050', duration=None):
    input_options = {"ac": 1}
    if duration is not None:
        input_options["t"] = duration
    process = (
        ffmpeg
        .input("pipe:", **input_options)
        .output("pipe:", format='wav',
                acodec='pcm_s16le',
                ar=target_ar,
                ac=1)  # Select WAV output format, and pcm_s16le auidio codec.
        .run_async(pipe_stdin=True, pipe_stdout=True)
    )
    audio_buffer = process.communicate(input=file.read())[0]
    process.wait()
    audio_io = BytesIO(audio_buffer)
    return audio_io


@st.cache_resource
def feature_plot(key: Any):
    embeddings = st.session_state["model_history"][key]["embeddings"]
    pca = st.session_state["model_history"][key]["pca"]
    scaler = st.session_state["model_history"][key]["scaler"]
    labels = st.session_state["model_history"][key]["labels_str"]
    genres = ('Blues', 'Classical', 'Country', 'Disco', 'Hiphop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock')

    proj = pca.transform(scaler.transform(embeddings))
    trues = labels
    fig = px.scatter_3d(
        x=proj[:, 0], y=proj[:, 1], z=proj[:, 2],
        color=trues,
        color_discrete_map={
            k: v for k, v in zip(genres, px.colors.qualitative.Set3)
        },
    )
    fig.update_traces(marker_size=4)
    return fig

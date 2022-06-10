# Example usage:
# python segment_laughter.py tst_wave.wav --output_dir=./tst_wave --min_length=0.2 --threshold=0.5
import argparse
import os
import pickle
import sys
import time
from distutils.util import strtobool
from functools import partial

import click
import librosa
import numpy as np
import pandas as pd
import scipy
import tgt
import torch
from torch import nn, optim
from tqdm import tqdm

from laughter_detector import configs
from laughter_detector import laugh_segmenter
from laughter_detector import models
from .utils import audio_utils, data_loaders, dataset_utils, torch_utils, s3_utils

SAMPLE_RATE = 8000


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    return device


def load_model(config, model_weights: str, storage_dir: str, device: torch.device("cuda")):
    model_path = s3_utils.maybe_download_from_aws(model_weights, storage_dir)
    model = config['model'](
        dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes']
    )
    model.set_device(device)

    if os.path.exists(model_path):
        torch_utils.load_checkpoint(model_path, model)
        return model.eval()
    else:
        raise Exception(f"Model checkpoint not found at {model_path}")


def create_loader(config, wav, orig_sr, target_sr, batch_size: int = 1, num_workers: int = 1):
    feature_fn = config['feature_fn']
    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        wav=wav, feature_fn=feature_fn, orig_sr=orig_sr, target_sr=target_sr
    )
    collate_fn = partial(audio_utils.pad_sequences_with_labels, expand_channel_dim=config['expand_channel_dim'])
    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return inference_generator


def predict(inference_generator: torch.utils.data.DataLoader, model: nn.Module, device: torch.device):
    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    return np.array(probs)


def cut_non_laughter(wav, probs, threshold, min_length, orig_sr: int = 44100):
    if len(probs) < 9:
        print("Number of probs is lower than the paddle length - to short signal")
        return []

    file_length = wav.shape[0] / orig_sr

    fps = len(probs) / float(file_length)
    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_clean_instances(probs, threshold=threshold, min_length=min_length, fps=fps)

    all_segments = []
    if len(instances) > 0:
        for index, instance in enumerate(instances):
            segments = laugh_segmenter.cut_segments([instance], wav, orig_sr)
            all_segments.append(segments)
    return all_segments


def extract_clean_instances(
    wav, orig_sr, model_weights: str, storage_dir: str, config: str, threshold: float, min_length: int
):
    config = configs.CONFIG_MAP[config]
    device = get_device()
    model = load_model(config, model_weights, storage_dir, device)
    # wav, sr = librosa.load(input_audio_file)
    loader = create_loader(config, wav, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
    probs = predict(loader, model, device)
    instances = cut_non_laughter(wav, probs, threshold, min_length)
    return instances


def write(wav, orig_sr, segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    maxv = np.iinfo(np.int16).max
    wav_paths = []
    # full_res_y, full_res_sr = librosa.load(audio_path, sr=44100)
    for index, s in enumerate(segments):
        wav_path = output_dir + "/laugh_" + str(index) + ".wav"
        scipy.io.wavfile.write(wav_path, orig_sr, (s * maxv).astype(np.int16))
        wav_paths.append(wav_path)
    print(laugh_segmenter.format_outputs(segments, wav_paths))


class LaughterRemover:
    def __init__(
        self,
        model_weights: str,
        storage_dir: str,
        config: str,
        threshold: float,
        min_length: int,
        batch_size: int = 1,
        num_workers: int = 1,
        orig_sr: int = 44100
    ):
        self.device = get_device()
        self.config = configs.CONFIG_MAP[config]
        self.model = load_model(self.config, model_weights, storage_dir, self.device)
        self.threshold = threshold
        self.min_length = min_length
        self.orig_sr = orig_sr
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(self, wav, cuda=True):
        loader = create_loader(
            self.config,
            wav,
            orig_sr=self.orig_sr,
            target_sr=SAMPLE_RATE,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
        print("Loader len: ", len(loader))
        probs = predict(loader, self.model, self.device)
        instances = cut_non_laughter(wav, probs, self.threshold, self.min_length, orig_sr=self.orig_sr)
        return instances


@click.command()
@click.argument('input_audio_file')
@click.option('--model_weights', type=str, default='resnet_with_augment.pt')
@click.option('--storage_dir', type=str, default=os.path.expanduser("~/.config/laughter_detector/"))
@click.option('--config', type=str, default='resnet_with_augmentation')
@click.option('--threshold', type=float, default=0.5)
@click.option('--min_length', type=float, default=0.2)
@click.option('--output_dir', type=str, default=None)
def main(
    input_audio_file: str, model_weights: str, storage_dir: str, config: str, threshold: float, min_length: int,
    output_dir: str
):
    wav, orig_sr = librosa.load(input_audio_file)
    remover = LaughterRemover(model_weights, storage_dir, config, threshold, min_length, orig_sr)
    segments = remover(wav)
    write(wav, orig_sr, segments, output_dir)


if __name__ == '__main__':
    main()

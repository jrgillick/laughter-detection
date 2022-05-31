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

import configs
import laugh_segmenter
import models
from utils import audio_utils, data_loaders, dataset_utils, torch_utils, s3_utils

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


def create_loader(config, audio_path, sample_rate):
    feature_fn = config['feature_fn']
    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate
    )
    collate_fn = partial(audio_utils.pad_sequences_with_labels, expand_channel_dim=config['expand_channel_dim'])
    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn
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


def cut_non_laughter(audio_path: str, probs, threshold, min_length):
    file_length = audio_utils.get_audio_length(audio_path)

    fps = len(probs) / float(file_length)
    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_clean_instances(probs, threshold=threshold, min_length=min_length, fps=fps)

    all_segments = []
    if len(instances) > 0:
        full_res_y, full_res_sr = librosa.load(audio_path, sr=44100)
        for index, instance in enumerate(instances):
            segments = laugh_segmenter.cut_segments([instance], full_res_y, full_res_sr)
            all_segments.append(segments)
    return all_segments


def extract_clean_instances(input_audio_file: str, model_weights: str, storage_dir: str, config: str, threshold: float, min_length: int):
    config = configs.CONFIG_MAP[config]
    device = get_device()
    model = load_model(config, model_weights, storage_dir, device)
    loader = create_loader(config, input_audio_file, SAMPLE_RATE)
    probs = predict(loader, model, device)
    instances = cut_non_laughter(input_audio_file, probs, threshold, min_length)
    return instances


def write(audio_path, segments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    maxv = np.iinfo(np.int16).max
    wav_paths = []
    full_res_y, full_res_sr = librosa.load(audio_path, sr=44100)
    for index, s in enumerate(segments):
        wav_path = output_dir + "/laugh_" + str(index) + ".wav"
        scipy.io.wavfile.write(wav_path, full_res_sr, (s * maxv).astype(np.int16))
        wav_paths.append(wav_path)
    print(laugh_segmenter.format_outputs(segments, wav_paths))


@click.command()
@click.argument('input_audio_file')
@click.option('--model_weights', type=str, default='resnet_with_augment.pt')
@click.option('--storage_dir', type=str, default=os.path.expanduser("~/.config/laughter_detector/"))
@click.option('--config', type=str, default='resnet_with_augmentation')
@click.option('--threshold', type=float, default=0.5)
@click.option('--min_length', type=float, default=0.2)
@click.option('--output_dir', type=str, default=None)
def main(input_audio_file: str, model_weights: str, storage_dir: str, config: str, threshold: float, min_length: int, output_dir: str):
    segments = extract_clean_instances(input_audio_file, model_weights, storage_dir, config, threshold, min_length)
    write(input_audio_file, segments, output_dir)


if __name__ == '__main__':
    main()

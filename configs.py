import sys, numpy as np
sys.path.append('./utils')
import models, audio_utils
from functools import partial

# takes a batch tuple (X,y)
def add_channel_dim(X):
    return np.expand_dims(X,1)

CONFIG_MAP = {}

CONFIG_MAP['mlp_mfcc'] = {
    'batch_size': 32,
    'model': models.MLPModel,
    'feature_fn': partial(audio_utils.featurize_mfcc, hop_length=186),
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': './data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': './data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': './data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': './data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'audioset_noisy_train_audio_pkl_path': './data/audioset/train/audioset_train_audios.pkl',
    'linear_layer_size': 44*40,
    'filter_sizes': None,
    'augment_fn': None,
    'expand_channel_dim': False,
    'supervised_augment': False,
    'supervised_spec_augment': False
}

CONFIG_MAP['resnet_base'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': './data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': './data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': './data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': './data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'audioset_noisy_train_audio_pkl_path': './data/audioset/train/audioset_train_audios.pkl',
    'augment_fn': None,
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'supervised_augment': False,
    'supervised_spec_augment': False
}

CONFIG_MAP['resnet_with_augmentation'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'val_data_text_path': './data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': './data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': './data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': './data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': './data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'audioset_noisy_train_audio_pkl_path': './data/audioset/train/audioset_train_audios.pkl',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
    'expand_channel_dim': True,
    'supervised_augment': True,
    'supervised_spec_augment': True
}
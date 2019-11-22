import sys, numpy as np
sys.path.append('/mnt/data0/jrgillick/projects/audio-feature-learning/')
sys.path.append('../')
import models, audio_utils
from functools import partial

# takes a batch tuple (X,y)
def add_channel_dim(X):
    return np.expand_dims(X,1)

CONFIG_MAP = {}

CONFIG_MAP['mlp_mfcc'] = {
    'batch_size': 500,
    'model': models.MLPModel,
    'feature_fn': audio_utils.featurize_mfcc,
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_0.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 100,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': None,
    'expand_channel_dim': False
}

CONFIG_MAP['resnet_melspec'] = {
    'batch_size': 128,
    'model': models.ResNet,
    'feature_fn': audio_utils.featurize_melspec,
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_1.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 100,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': None,
    'expand_channel_dim': True
}

CONFIG_MAP['resnet_melspec_bigger'] = {
    'batch_size': 128,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=80),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 100,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': None,
    'expand_channel_dim': True
}

CONFIG_MAP['resnet_melspec_bigger_augment'] = {
    'batch_size': 128,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=80),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_4.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 100,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'expand_channel_dim': True
}

CONFIG_MAP['resnet_melspec_bigger_43fps'] = {
    'batch_size': 128,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': None,
    'linear_layer_size': 64,
    'expand_channel_dim': True
}
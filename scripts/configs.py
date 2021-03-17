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
    'feature_fn': partial(audio_utils.featurize_mfcc, hop_length=80),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_0.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 100,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'linear_layer_size': 101*40,
    'filter_sizes': None,
    'augment_fn': None,
    'expand_channel_dim': False
}

CONFIG_MAP['mlp_mfcc_43fps'] = {
    'batch_size': 32,
    'model': models.MLPModel,
    'feature_fn': partial(audio_utils.featurize_mfcc, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 100,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'audioset_noisy_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'linear_layer_size': 44*40,
    'filter_sizes': None,
    'augment_fn': None,
    'expand_channel_dim': False,
    'supervised_augment': False,
    'supervised_spec_augment': False,
    'unsupervised_spec_augment': False
}

CONFIG_MAP['resnet_melspec_bigger_43fps'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': None,
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'supervised_augment': False,
    'supervised_spec_augment': True,
    'unsupervised_spec_augment': False
}

CONFIG_MAP['resnet_base_bigger_43fps'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'audioset_noisy_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'augment_fn': None,
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'supervised_augment': False,
    'supervised_spec_augment': False,
    'unsupervised_spec_augment': False
}

CONFIG_MAP['resnet_43fps_spec_augment'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'supervised_augment': False,
    'supervised_spec_augment': True,
    'unsupervised_spec_augment': False
}

CONFIG_MAP['resnet_43fps_wav_augment_spec_augment'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'supervised_augment': True,
    'supervised_spec_augment': True,
    'unsupervised_spec_augment': False
}

CONFIG_MAP['resnet_43fps_spec_augment_large'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
    'expand_channel_dim': True,
    'supervised_augment': False,
    'supervised_spec_augment': True,
    'unsupervised_spec_augment': False
}

CONFIG_MAP['resnet_43fps_wav_augment_spec_augment_large'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'audioset_noisy_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
    'expand_channel_dim': True,
    'supervised_augment': True,
    'supervised_spec_augment': True,
    'unsupervised_spec_augment': False
}

##### FIXMATCH ######
CONFIG_MAP['fixmatch_resnet_43fps_wav_augment_spec_augment_large'] = {
    'batch_size': 16,
    'consistency_batch_size': 80,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment_strong, sr=8000),
    'linear_layer_size': 128,
    'filter_sizes': [128,64,32,32],
    'expand_channel_dim': True,
    'supervised_augment': True,
    'supervised_spec_augment': True,
    'consistency_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'consistency_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/val/audioset_val_audios.pkl',
}


################  UDA Training #################
CONFIG_MAP['consistency_mlp_mfcc_43fps'] = {
    'batch_size': 128,
    'consistency_batch_size': 128,
    'model': models.MLPModel,
    'feature_fn': partial(audio_utils.featurize_mfcc, sr=8000, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_0.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 100,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'linear_layer_size': 44*40,
    'filter_sizes': [64,32,16,16],
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'expand_channel_dim': False,
    'consistency_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'consistency_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/val/audioset_val_audios.pkl',
    'supervised_augment': False,
    'supervised_spec_augment': False,
    'unsupervised_spec_augment': False
}

CONFIG_MAP['consistency_resnet_melspec_bigger_43fps'] = {
    'batch_size': 128,
    'consistency_batch_size': 128,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'consistency_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'consistency_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/val/audioset_val_audios.pkl',
    'supervised_augment': False,
    'supervised_spec_augment': False,
    'unsupervised_spec_augment': False
}

CONFIG_MAP['consistency_resnet_melspec_bigger_43fps_spec_aug'] = {
    'batch_size': 128,
    'consistency_batch_size': 128,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'consistency_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'consistency_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/val/audioset_val_audios.pkl',
    'supervised_augment': False,
    'supervised_spec_augment': False,
    'unsupervised_spec_augment': True
}

CONFIG_MAP['consistency_resnet_43fps_aug'] = {
    'batch_size': 8,
    'consistency_batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/switchboard_val_data.txt',
    'log_frequency': 200,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'consistency_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'consistency_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/val/audioset_val_audios.pkl',
    'supervised_augment': True,
    'supervised_spec_augment': True,
    'unsupervised_spec_augment': True
}

CONFIG_MAP['consistency_resnet_43fps_spec_aug_few_examples'] = {
    'batch_size': 4,
    'consistency_batch_size': 256,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 100,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'consistency_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl',
    'consistency_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/val/audioset_val_audios.pkl',
    'supervised_augment': True,
    'supervised_spec_augment': True,
    'unsupervised_spec_augment': True
}

CONFIG_MAP['resnet_43fps_spec_aug_few_examples'] = {
    'batch_size': 32,
    'model': models.ResNetBigger,
    'feature_fn': partial(audio_utils.featurize_melspec, hop_length=186),
    'train_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/train_3.txt',
    'val_data_text_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/val.txt',
    'log_frequency': 10,
    'swb_train_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/train/swb_train_audios.pkl',
    'swb_val_audio_pkl_path': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/val/swb_val_audios.pkl',
    'swb_audio_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/',
    'swb_transcription_root': '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/',
    'augment_fn': partial(audio_utils.random_augment, sr=8000),
    'linear_layer_size': 64,
    'filter_sizes': [64,32,16,16],
    'expand_channel_dim': True,
    'supervised_augment': True,
    'supervised_spec_augment': True,
    'unsupervised_spec_augment': False
}

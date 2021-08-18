# Usage
# `python store_all_switchboard_audio.py`
# This assumes you'll have downloaded switchboard data into a specific location.
# And also assumes you'll be (outputting) pre-processed data into another specific location.
# If you want to change the paths, adjust them in the strings below.

import sys, time, librosa, os, pickle
sys.path.append('../utils/')
from tqdm import tqdm
import dataset_utils, audio_utils, data_loaders
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

#switchboard_audio_path
a_root = '../data/switchboard/switchboard-1/97S62/' 

#switchboard_transcriptions_path
t_root = '../data/switchboard/switchboard-1/swb_ms98_transcriptions/'


all_audio_files = librosa.util.find_files(a_root,ext='sph')
train_folders, val_folders, test_folders = dataset_utils.get_train_val_test_folders(t_root)

train_transcription_files_A, train_audio_files = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(train_folders, 'A'), all_audio_files)
train_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(train_folders, 'B'), all_audio_files)

val_transcription_files_A, val_audio_files = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(val_folders, 'A'), all_audio_files)
val_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(val_folders, 'B'), all_audio_files)

test_transcription_files_A, test_audio_files = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(test_folders, 'A'), all_audio_files)
test_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(
    dataset_utils.get_all_transcriptions_files(test_folders, 'B'), all_audio_files)

h = {}
train_y = audio_utils.parallel_load_audio_batch(train_audio_files, n_processes=8, sr=8000)
assert(len(train_y) == len(train_audio_files))
for i in range(len(train_audio_files)):
    f = train_audio_files[i]
    y = train_y[i]
    h[f] = y

with open("../data/switchboard/train/swb_train_audios1.pkl", "wb") as f:
    pickle.dump(h, f)


h = {}
val_y = audio_utils.parallel_load_audio_batch(val_audio_files, n_processes=8, sr=8000)
assert(len(val_y) == len(val_audio_files))
for i in range(len(val_audio_files)):
    f = val_audio_files[i]
    y = val_y[i]
    h[f] = y

with open("../data/switchboard/val/swb_val_audios1.pkl", "wb") as f:
    pickle.dump(h, f)


h = {}
test_y = audio_utils.parallel_load_audio_batch(test_audio_files, n_processes=8, sr=8000)
assert(len(test_y) == len(test_audio_files))
for i in range(len(test_audio_files)):
    f = test_audio_files[i]
    y = test_y[i]
    h[f] = y

with open("../data/switchboard/test/swb_test_audios1.pkl", "wb") as f:
    pickle.dump(h, f)
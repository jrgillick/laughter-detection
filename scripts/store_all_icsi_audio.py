# Usage
# `python store_all_icsi_audio.py`
# This assumes you'll have downloaded icsi data into a specific location.
# And also assumes you'll be (outputting) pre-processed data into another specific location.
# If you want to change the paths, adjust them in the strings below.

import sys
import librosa
import os
import pickle
sys.path.append('../utils/')
import audio_utils
import dataset_utils

# absolute icsi_audio_path (make sure it includes trailing slash)
a_root = os.path.realpath('../data/icsi/Signals')

# absolute icsi_transcriptions_path (make sure it includes trailing slash)
t_root = os.path.realpath('../data/icsi/transcripts/')

# icsi sample rate
sample_rate = 16000

# all_audio_files = librosa.util.find_files(a_root, ext='sph')
train_folders, val_folders, test_folders = dataset_utils.get_train_val_test_folders(
    a_root)

# Get audio files for each split
print(a_root)
print(train_folders)
train_audio_files = [f'{dir}/{f}' for dir in train_folders for f in os.listdir(dir)]
val_audio_files = [f'{dir}/{f}' for dir in val_folders for f in os.listdir(dir)]
test_audio_files = [f'{dir}/{f}' for dir in test_folders for f in os.listdir(dir)]


# CREATE TRAINING HASHMAP
h = {}
train_y = audio_utils.parallel_load_audio_batch(
    train_audio_files, n_processes=8, sr=sample_rate)
# Make sure all files have been loaded
assert(len(train_y) == len(train_audio_files))
for i in range(len(train_audio_files)):
    f = train_audio_files[i]
    y = train_y[i]
    h[f] = y

with open("../data/icsi/train/swb_train_audios1.pkl", "wb") as f:
    pickle.dump(h, f)


# CREATE VALIDATION HASHMAP
h = {}
val_y = audio_utils.parallel_load_audio_batch(
    val_audio_files, n_processes=8, sr=sample_rate)
# Make sure all files have been loaded
assert(len(val_y) == len(val_audio_files)) 

for i in range(len(val_audio_files)):
    f = val_audio_files[i]
    y = val_y[i]
    h[f] = y

with open("../data/icsi/val/swb_val_audios1.pkl", "wb") as f:
    pickle.dump(h, f)


# CREATE TEST HASHMAP
h = {}
test_y = audio_utils.parallel_load_audio_batch(
    test_audio_files, n_processes=8, sr=sample_rate)
# Make sure all files have been loaded
assert(len(test_y) == len(test_audio_files))
for i in range(len(test_audio_files)):
    f = test_audio_files[i]
    y = test_y[i]
    h[f] = y

with open("../data/icsi/test/swb_test_audios1.pkl", "wb") as f:
    pickle.dump(h, f)

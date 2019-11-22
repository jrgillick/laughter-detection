from audio_set_loading import *
import sys, time, librosa, os, argparse, pickle
sys.path.append('/home/jrgillick/projects/audio-feature-learning/')
sys.path.append('/mnt/data0/jrgillick/projects/audio-feature-learning/')
from tqdm import tqdm
import dataset_utils, audio_utils, data_loaders
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# train
print("Loading train files...")
h = {}
train_y = audio_utils.parallel_load_audio_batch(audioset_train_files, n_processes=8, sr=8000)
assert(len(train_y) == len(audioset_train_files))
for i in range(len(audioset_train_files)):
    f = audioset_train_files[i]
    y = train_y[i]
    h[f] = y
    
with open("/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/train/audioset_train_audios.pkl", "wb") as f:
    pickle.dump(h, f)

# val
print("Loading val files...")
h = {}
val_y = audio_utils.parallel_load_audio_batch(audioset_val_files, n_processes=8, sr=8000)
assert(len(val_y) == len(audioset_val_files))
for i in range(len(audioset_val_files)):
    f = audioset_val_files[i]
    y = val_y[i]
    h[f] = y
    
with open("/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/val/audioset_val_audios.pkl", "wb") as f:
    pickle.dump(h, f)

# test
print("Loading test files...")
h = {}
test_y = audio_utils.parallel_load_audio_batch(audioset_test_files, n_processes=8, sr=8000)
assert(len(test_y) == len(audioset_test_files))
for i in range(len(audioset_test_files)):
    f = audioset_test_files[i]
    y = test_y[i]
    h[f] = y
    
with open("/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/test/audioset_test_audios.pkl", "wb") as f:
    pickle.dump(h, f)
    
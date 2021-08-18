from audio_set_loading import *
import sys, time, librosa, os, argparse, pickle
from tqdm import tqdm
sys.path.append('../utils/')
import dataset_utils, audio_utils, data_loaders
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# val
print("Loading val files...")
h = {}
val_y = audio_utils.parallel_load_audio_batch(audioset_dev_files, n_processes=8, sr=8000)
assert(len(val_y) == len(audioset_dev_files))
for i in range(len(audioset_dev_files)):
    f = audioset_dev_files[i]
    y = val_y[i]
    h[f] = y
    
with open("../data/audioset/val/audioset_val_audios.pkl", "wb") as f:
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
    
with open("../data/audioset/test/audioset_test_audios.pkl", "wb") as f:
    pickle.dump(h, f)
    
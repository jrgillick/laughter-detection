import pandas as pd
import numpy as np, os, sys, librosa
import warnings
warnings.simplefilter("ignore")

MIN_GAP = 0
avoid_edges=True
edge_gap = 0.5

# Predict w/ pytorch code for audioset data
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../utils/')
import models, configs, torch
import dataset_utils, audio_utils, data_loaders, torch_utils
from torch import optim, nn
device = torch.device('cpu')
from eval_utils import *
warnings.simplefilter("ignore")
from tqdm import tqdm

config = configs.CONFIG_MAP['mlp_mfcc']
model = config['model'](linear_layer_size=config['linear_layer_size'])
model.set_device(device)

checkpoint_dir = '../../checkpoints/comparisons/baseline_mfcc_trained_on_audioset'

if os.path.exists(checkpoint_dir):
    torch_utils.load_checkpoint(checkpoint_dir+'/best.pth.tar', model)
else:
    print("Checkpoint not found")
    
model.eval()

audioset_annotations_df = pd.read_csv('../../data/audioset/annotations/clean_laughter_annotations.csv')
print("\nAudioset Annotations stats:")
total_audioset_minutes, total_audioset_laughter_minutes, total_audioset_non_laughter_minutes, audioset_laughter_fraction, audioset_laughter_count = get_annotation_stats(
    audioset_annotations_df, display=True, min_gap = MIN_GAP, avoid_edges=True, edge_gap=0.5)

audioset_distractor_df = pd.read_csv('../../data/audioset/annotations/clean_distractor_annotations.csv')
audioset_annotations_df = pd.concat([audioset_annotations_df, audioset_distractor_df])
audioset_annotations_df.reset_index(inplace=True, drop=True)
    
all_results = []
for index in tqdm(range(len(audioset_annotations_df))):
    line = audioset_annotations_df.iloc[index]
    h = get_results_for_annotation_index(model, config, audioset_annotations_df, index, min_gap=0.,
                                         threshold=0.5, use_filter=False, min_length=0.0,
                                         avoid_edges=True, edge_gap=0.5)
    all_results.append(h)

results_df = pd.DataFrame(all_results)
results_df.to_csv("noisy_audioset_trained_baseline_audioset_results.csv",index=None)

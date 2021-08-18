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
model = config['model'](dropout_rate=0.5, linear_layer_size=config['linear_layer_size'])
model.set_device(device)
model.to(device)

checkpoint_dir = '../../checkpoints/comparisons/baseline_mfcc'

if os.path.exists(checkpoint_dir):
    torch_utils.load_checkpoint(checkpoint_dir+'/best.pth.tar', model)
else:
    print("Checkpoint not found")
    
model.eval()
    

swb_val_distractor_df = pd.read_csv('../../data/switchboard/annotations/clean_switchboard_val_distractor_annotations.csv')
swb_test_distractor_df = pd.read_csv('../../data/switchboard/annotations/clean_switchboard_test_distractor_annotations.csv')
    
swb_val_df = pd.read_csv('../../data/switchboard/annotations/clean_switchboard_val_laughter_annotations.csv')
print("\nSwitchboard Val Annotations stats:")
_, _, _, _, _ = get_annotation_stats(
    swb_val_df, display=True, min_gap = MIN_GAP, avoid_edges=True, edge_gap=0.5)

swb_val_df = pd.concat([swb_val_df, swb_val_distractor_df])
swb_val_df.reset_index(inplace=True, drop=True)

    
swb_val_results = []
for index in tqdm(range(len(swb_val_df))):
    line = swb_val_df.iloc[index]
    h = get_results_for_annotation_index(model, config, swb_val_df, index, min_gap=0.,
                                         threshold=0.5, use_filter=False, min_length=0.0,
                                         avoid_edges=True, edge_gap=0.5)
    swb_val_results.append(h)

val_results_df = pd.DataFrame(swb_val_results)
val_results_df.to_csv("baseline_switchboard_val_results.csv",index=None)


swb_test_df = pd.read_csv('../../data/switchboard/annotations/clean_switchboard_test_laughter_annotations.csv')
print("\nSwitchboard Test Annotations stats:")
_, _, _, _, _ = get_annotation_stats(
    swb_test_df, display=True, min_gap = MIN_GAP, avoid_edges=True, edge_gap=0.5)

swb_test_df = pd.concat([swb_test_df, swb_test_distractor_df])
swb_test_df.reset_index(inplace=True, drop=True)
    
swb_test_results = []
for index in tqdm(range(len(swb_test_df))):
    line = swb_test_df.iloc[index]
    h = get_results_for_annotation_index(model, config, swb_test_df, index, min_gap=0.,
                                         threshold=0.5, use_filter=False, min_length=0.0,
                                         avoid_edges=True, edge_gap=0.5)
    swb_test_results.append(h)

test_results_df = pd.DataFrame(swb_test_results)
test_results_df.to_csv("baseline_switchboard_test_results.csv",index=None)

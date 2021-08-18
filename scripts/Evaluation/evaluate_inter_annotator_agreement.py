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


audioset_annotations_df = pd.read_csv('../../data/audioset/annotations/clean_laughter_annotations.csv')
audioset_annotations2_df = pd.read_csv('../../data/audioset/annotations/clean_2nd_annotator_annotations.csv')

def get_inter_annotator_for_ID(audioset_ID, annotations_df, annotations_df2, min_gap=0.,
                               threshold=0.5, use_filter=False, min_length=0.0,
                               avoid_edges=True, edge_gap=0.5, expand_channel_dim=False):
    
    annotations1_index = annotations_df[annotations_df['FileID'] == audioset_ID].index[0]
    annotations1_line = dict(annotations_df.iloc[annotations1_index])
    
    true_laughter_times = get_laughter_times_from_annotation_line(
            annotations1_line,min_gap=min_gap,avoid_edges=avoid_edges,edge_gap=edge_gap)
    true_non_laughter_times = get_non_laughter_times(
            true_laughter_times,annotations1_line['window_start'],annotations1_line['window_length'],avoid_edges=avoid_edges,edge_gap=edge_gap)
    
    annotations2_index = annotations_df2[annotations_df2['FileID'] == audioset_ID].index[0]
    annotations2_line = dict(annotations_df2.iloc[annotations2_index])
    
    predicted_laughter_times = get_laughter_times_from_annotation_line(
            annotations2_line,min_gap=min_gap,avoid_edges=avoid_edges,edge_gap=edge_gap)
    predicted_non_laughter_times = get_non_laughter_times(
            predicted_laughter_times,annotations2_line['window_start'],annotations2_line['window_length'],avoid_edges=avoid_edges,edge_gap=edge_gap)
     
    total_laughter_time = sum_overlap_amount(true_laughter_times,true_laughter_times)
    total_non_laughter_time = sum_overlap_amount(true_non_laughter_times,true_non_laughter_times)

    true_positive_time = sum_overlap_amount(true_laughter_times, predicted_laughter_times)
    true_negative_time = sum_overlap_amount(true_non_laughter_times, predicted_non_laughter_times)
    false_positive_time = sum_overlap_amount(true_non_laughter_times, predicted_laughter_times)
    false_negative_time = sum_overlap_amount(true_laughter_times, predicted_non_laughter_times)
    
    total_time = true_positive_time + true_negative_time + false_positive_time + false_negative_time
    
    try:
        assert(np.abs(total_laughter_time - (true_positive_time + false_negative_time)) < 0.2)
        assert(np.abs(total_non_laughter_time - (true_negative_time + false_positive_time)) < 0.2)
    except:
        print(audioset_ID)
        print(annotations1_line['window_length'])
        print(np.abs(total_laughter_time - (true_positive_time + false_negative_time)))
        print("\n") 
    
    h = {'FileID':audioset_ID, 'tp_time':true_positive_time, 'tn_time':true_negative_time,
         'fp_time':false_positive_time, 'fn_time':false_negative_time,
         'predicted_laughter': predicted_laughter_times, 'predicted_non_laughter': predicted_non_laughter_times,
         'true_laughter': true_laughter_times, 'true_non_laughter': true_non_laughter_times}

    return h


double_annotated_ids = list(set(audioset_annotations_df.FileID) & set(audioset_annotations2_df.FileID))

all_results = []
for audioset_ID in tqdm(double_annotated_ids):
    h = get_inter_annotator_for_ID(audioset_ID, audioset_annotations_df, audioset_annotations2_df,
                                   min_gap=0., threshold=0.5, use_filter=False, min_length=0.0,
                                   avoid_edges=True, edge_gap=0.5)
    all_results.append(h)

results_df = pd.DataFrame(all_results)
results_df.to_csv("interannotator_agreement_results.csv",index=None)
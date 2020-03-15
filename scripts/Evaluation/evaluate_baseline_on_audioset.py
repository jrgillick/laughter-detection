import pandas as pd
import numpy as np, os, sys
import warnings
warnings.simplefilter("ignore")
from eval_utils import *
from tqdm import tqdm

path_to_baseline_code = '/mnt/data0/jrgillick/projects/laughter-detection-2018/laughter-detection/'
baseline_model_path = path_to_baseline_code + '/models/model.h5'
sys.path.append(path_to_baseline_code)
import laugh_segmenter as baseline_laugh_segmenter


annotations_df = pd.read_csv('../../data/audioset/annotations/clean_laughter_annotations.csv')



def get_baseline_results_per_annotation_index(model, annotations_df, i):
    audio_file = annotations_df.audio_path.iloc[i]

    true_laughter_times = get_laughter_times_from_annotation_line(dict(annotations_df.iloc[i]))
    true_non_laughter_times = get_non_laughter_times(true_laughter_times, annotations_df.iloc[i].audio_length)

    predicted_laughter_times = baseline_laugh_segmenter.segment_laugh_with_model(
        model, input_path=audio_file, threshold=0.5, min_length=0.1)

    predicted_non_laughter_times = get_non_laughter_times(predicted_laughter_times, annotations_df.iloc[i].audio_length)

    total_laughter_time = sum_overlap_amount(true_laughter_times,true_laughter_times)
    total_non_laughter_time = sum_overlap_amount(true_non_laughter_times,true_non_laughter_times)

    true_positive_time = sum_overlap_amount(true_laughter_times, predicted_laughter_times)
    true_negative_time = sum_overlap_amount(true_non_laughter_times, predicted_non_laughter_times)
    false_positive_time = sum_overlap_amount(true_non_laughter_times, predicted_laughter_times)
    false_negative_time = sum_overlap_amount(true_laughter_times, predicted_non_laughter_times)
    
    total_time = true_positive_time + true_negative_time + false_positive_time + false_negative_time
    
    assert(np.abs(total_laughter_time - (true_positive_time + false_negative_time)) < 0.1)
    assert(np.abs(total_non_laughter_time - (true_negative_time + false_positive_time)) < 0.1)
    assert(np.abs(total_time - annotations_df.iloc[i].audio_length) < 0.1)
    assert(np.abs(total_time - (total_laughter_time + total_non_laughter_time)) < 0.1)

    h = {'FileID':annotations_df.FileID[i], 'tp_time':true_positive_time, 'tn_time':true_negative_time,
         'fp_time':false_positive_time, 'fn_time':false_negative_time,
         'predicted_laughter': predicted_laughter_times, 'predicted_non_laughter': predicted_non_laughter_times}

    return h

baseline_model = baseline_laugh_segmenter.load_model(baseline_model_path)

all_results = []
for i in tqdm(range(len(annotations_df))):
    h = get_baseline_results_per_annotation_index(baseline_model, annotations_df, i)
    all_results.append(h)
    
results_df = pd.DataFrame(all_results)
results_df.to_csv("baseline_audioset_results.csv",index=None)
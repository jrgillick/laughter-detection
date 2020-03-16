import sys, time, librosa, os, argparse, pickle, numpy as np
sys.path.append('./Evaluation')
from eval_utils import *
warnings.simplefilter("ignore")
from tqdm import tqdm
sys.path.append('/mnt/data0/jrgillick/projects/audio-feature-learning/')
import dataset_utils, audio_utils, data_loaders
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.utils import shuffle


MIN_GAP = 0.

# Load trained baseline model
path_to_baseline_code = '/mnt/data0/jrgillick/projects/laughter-detection-2018/laughter-detection/'
baseline_model_path = path_to_baseline_code + '/models/model.h5'
sys.path.append(path_to_baseline_code)
import laugh_segmenter as baseline_laugh_segmenter
baseline_model = baseline_laugh_segmenter.load_model(baseline_model_path)

# Load Switchboard data
t_root = '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/swb_ms98_transcriptions/'
a_root = '/mnt/data0/jrgillick/projects/laughter-detection/data/switchboard/switchboard-1/97S62/'
all_audio_files = librosa.util.find_files(a_root,ext='sph')
train_folders, val_folders, test_folders = dataset_utils.get_train_val_test_folders(t_root)

train_transcription_files_A, train_audio_files = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(train_folders, 'A'), all_audio_files)
train_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(train_folders, 'B'), all_audio_files)

val_transcription_files_A, val_audio_files = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(val_folders, 'A'), all_audio_files)
val_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(val_folders, 'B'), all_audio_files)

test_transcription_files_A, test_audio_files = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(test_folders, 'A'), all_audio_files)
test_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(test_folders, 'B'), all_audio_files)


# Get stats on the Laughter/Non-Laughter classes in the Audioset Annotations
# So that we can use these stats to resample from the switchboard test set
# Sample the Switchboard test data to be roughly proportional to what we have from the Audioset annotations
# in terms of class balance (~34.6% laughter vs. ~65.4% non-laughter)
# Do this by extending the audio files in each direction for the annotated laughter
annotations_df = pd.read_csv('../../data/audioset/annotations/clean_laughter_annotations.csv')

print("\nAudioset Annotations stats:")
total_audioset_minutes, total_audioset_laughter_minutes, total_audioset_non_laughter_minutes, audioset_laughter_fraction, audioset_laughter_count = get_annotation_stats(annotations_df, display=True, min_gap = MIN_GAP)


# Make a dataframe that matches the structure of the Audioset annotations
# Extend the audio files in each direction around the annotated laughter to get the "non-laughter"
def make_switchboard_dataframe(transcription_files_A, transcription_files_B, audio_files, min_gap=MIN_GAP):
    rows = []
    for i in tqdm(range(len(transcription_files_A))):
        text_A = transcription_files_A[i]; text_B = transcription_files_B[i]
        audio_file = audio_files[i]
        switchboard_file_length = get_audio_file_length(audio_file)
        laughter_regions, speech_regions, _, _ = dataset_utils.get_laughter_regions_and_speech_regions(
            text_A, text_B, audio_file)
        for region in laughter_regions:
            # First add the necessary padding to laughter regions from the windowing model (min_gap)
            region_start = region[0]; region_length = region[1]
            temp_extra_length_per_side = min_gap + ((region_length / audioset_laughter_fraction) - region_length)/2
            
            
            region_start = np.maximum(0,region_start-min_gap)
            region_length = np.minimum(switchboard_file_length - region_start, region_length + (2*min_gap))
            region_end = region_start + region_length
            
            temp_start_time = np.maximum(region_start-temp_extra_length_per_side, 0.)
            temp_end_time = np.minimum(region_end+temp_extra_length_per_side,switchboard_file_length)
            
            extra_beginning_time = region_start-temp_start_time
            extra_end_time = temp_end_time-region_end
            total_time = region_length + extra_beginning_time + extra_end_time
            
            h = {'FileID': audio_file.split('/')[-1].split('.')[0],
                 'Start': region_start,
                 'End': region_end,
                 'Start.1':np.nan,'Start.2':np.nan,'Start.3':np.nan,'Start.4':np.nan, # match audioset annotations
                 'End.1':np.nan,'End.2':np.nan,'End.3':np.nan,'End.4':np.nan, # match audioset annotations
                 'audio_path': audio_file,
                 'audio_length': switchboard_file_length,
                 'extra_beginning_time': extra_beginning_time,
                 'extra_end_time': extra_end_time
                }
            rows.append(h)
    switchboard_annotations_df = pd.DataFrame(rows)
    return switchboard_annotations_df

print("\nSetting up SWB Validation data")
# Get results on SWB Validation Set
switchboard_val_annotations_df = make_switchboard_dataframe(
    val_transcription_files_A, val_transcription_files_B, val_audio_files, min_gap=MIN_GAP)

print("\nSwitchboard Validation Set Annotations Stats:")
val_swb_minutes, val_swb_laughter_minutes, val_swb_non_laughter_minutes, val_laughter_fraction, val_laughter_count = get_annotation_stats(switchboard_val_annotations_df, display=True, min_gap=MIN_GAP)

print("\nPredicting on Switchboard Validation Set...")
val_results = []
for i in tqdm(range(len(switchboard_val_annotations_df))):
    h = get_baseline_results_per_annotation_index(
        baseline_model, switchboard_val_annotations_df, baseline_laugh_segmenter, i, min_gap=MIN_GAP,
        threshold=0.5,use_filter=False, min_length=0.)
    val_results.append(h)
    
val_results_df = pd.DataFrame(val_results)
val_results_df.to_csv("baseline_switchboard_val_results.csv",index=None)

"""

print("Setting up SWB Test data")
# Get results on SWB Test Set
switchboard_test_annotations_df = make_switchboard_dataframe(
    test_transcription_files_A, test_transcription_files_B, test_audio_files)

print("Switchboard Test Set Annotations Stats:")
test_swb_minutes, test_swb_laughter_minutes, test_swb_non_laughter_minutes, test_laughter_fraction, test_laughter_count = get_annotation_stats(switchboard_test_annotations_df, display=True)

print("\nPredicting on Switchboard Test Set...")
test_results = []
for i in tqdm(range(len(switchboard_test_annotations_df))):
    h = get_baseline_results_per_annotation_index(
        baseline_model,switchboard_test_annotations_df, baseline_laugh_segmenter, i)
    test_results.append(h)
    
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv("baseline_switchboard_test_results.csv",index=None)
"""



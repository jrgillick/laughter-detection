import sys, time, librosa, os, argparse, pickle, numpy as np
sys.path.append('../')
sys.path.append('../utils/')
import dataset_utils, audio_utils, data_loaders
sys.path.append('./Evaluation')
from eval_utils import *
warnings.simplefilter("ignore")
from tqdm import tqdm


import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.utils import shuffle

MIN_GAP=0. # Only use if we want to enforce gap between annotations

# Load Switchboard data
t_root = '../data/switchboard/switchboard-1/swb_ms98_transcriptions/'
a_root = '../data/switchboard/switchboard-1/97S62/'
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
# in terms of class balance (~39.3% laughter)
# Do this by extending the audio files in each direction for the annotated laughter
audioset_annotations_df = pd.read_csv('../data/audioset/annotations/clean_laughter_annotations.csv')

print("\nAudioset Annotations stats:")
total_audioset_minutes, total_audioset_laughter_minutes, total_audioset_non_laughter_minutes, audioset_laughter_fraction, audioset_laughter_count = get_annotation_stats(audioset_annotations_df, display=True, min_gap = MIN_GAP, avoid_edges=True, edge_gap=0.5)

import torch
from torch import optim, nn

class AllocationModel(nn.Module):
    def __init__(self, total_time, available_time_per_clip):
        super().__init__()
        
        self.total_time = total_time
        self.available_time_per_clip = available_time_per_clip
        self.allocated_times = nn.Parameter(torch.zeros(len(available_time_per_clip)), requires_grad=True)
        
    def forward(self):
        total_time_used = torch.sum(self.allocated_times)
        leftover_times = self.available_time_per_clip - self.allocated_times
        total_time_loss = torch.abs(self.total_time - total_time_used)
        min_time = torch.min(leftover_times)
        leftover_loss = 1/(min_time)
        loss = total_time_loss + leftover_loss
        self.allocated_times.data = torch.clamp(self.allocated_times, 0)
        self.allocated_times.data = torch.min(self.allocated_times, self.available_time_per_clip)
        return loss, self.allocated_times   
    
def distribute_time(total_time, available_time_per_clip):
    am = AllocationModel(torch.tensor(total_time), torch.tensor(available_time_per_clip))
    optimizer = optim.Adam(am.parameters(),lr=0.1)
    for i in range(300):
        optimizer.zero_grad()
        loss, times = am()
        loss.backward(retain_graph=True)
        optimizer.step()
        am.zero_grad()
        #import pdb; pdb.set_trace()
    times = torch.clamp(times, 0)
    return np.nan_to_num(times.detach().cpu().numpy())

def get_10_second_clips(regions_list, audio_file_path, full_audio_file_length,
                        index, audioset_laughter_fraction, adjustment_amount=0):
    if len(regions_list) == 0:
        return [],[],0,0
    # First pass to find clips
    all_clips = []
    current_start = None; current_end = None
    for i in range(len(regions_list)):
        if current_start is None:
            current_start = regions_list[i][0]
            beginning_space = current_start
        if regions_list[i][0] + regions_list[i][1] > current_start + 10:
            all_clips.append({'window': [current_start, current_end],
                              'beginning_buffer':0., 'end_buffer':0.,
                              'beginning_space': beginning_space})
            current_start = regions_list[i][0]
            beginning_space = current_start - current_end # new start point to old end point
        current_end = regions_list[i][0] + regions_list[i][1]
    if current_start is not None and current_end is not None:
        end_space = full_audio_file_length - current_end
        all_clips.append({'window': [current_start, current_end],
                          'beginning_buffer':0., 'end_buffer':0.,
                          'beginning_space': beginning_space,
                          'end_space': end_space})
        
    for i, clip in enumerate(all_clips):
        if 'end_space' not in clip: clip['end_space'] = all_clips[i+1]['beginning_space']
        #clip_len = clip['window'][1] - clip['window'][0]
        
    # 2nd pass: Go through, extending by 0.5 secs on each side unless it exceeds 10 seconds
    for i, clip in enumerate(all_clips):
        start, end = clip['window']
        length = end-start
        # Try adding 0.5s to begin and end, if not possible, print and give up
        time_to_add_per_side = 0.5
        # Try adding to beginning and end
        if time_to_add_per_side < clip['beginning_space'] and time_to_add_per_side < clip['end_space']:
            clip['window'] = [start-time_to_add_per_side, end + time_to_add_per_side]
            clip['beginning_space'] -= time_to_add_per_side; clip['end_space'] -= time_to_add_per_side
            clip['beginning_buffer'] += 0.5; clip['end_buffer'] += 0.5
            if i > 0:
                all_clips[i-1]['end_space'] -= time_to_add_per_side
            if i < len(all_clips) - 1:
                all_clips[i+1]['beginning_space'] -= time_to_add_per_side
        
    # 3rd pass: Go back through, centering and extending windows out to 10s
    for i, clip in enumerate(all_clips):
        start, end = clip['window']
        length = end-start
        # Try adding equally to begin and end, if not possible, try one side, if not possible, print and give up
        time_to_add = np.maximum(10 - length, 0) # If longer than 10 secs, don't shorten it, just leave it
        time_to_add_per_side = time_to_add / 2
        # Try adding to beginning and end
        if time_to_add_per_side < clip['beginning_space'] and time_to_add_per_side < clip['end_space']:
            clip['window'] = [start-time_to_add_per_side, end + time_to_add_per_side]
            clip['beginning_space'] -= time_to_add_per_side; clip['end_space'] -= time_to_add_per_side
            clip['beginning_buffer'] += time_to_add_per_side; clip['end_buffer'] += time_to_add_per_side
            if i > 0:
                all_clips[i-1]['end_space'] -= time_to_add_per_side
            if i < len(all_clips) - 1:
                all_clips[i+1]['beginning_space'] -= time_to_add_per_side
        elif time_to_add < clip['beginning_space']:
            clip['window'] = [start-time_to_add, end]
            clip['beginning_buffer'] += time_to_add
            if i > 0:
                all_clips[i-1]['end_space'] -= time_to_add
        elif time_to_add < clip['end_space']:
            clip['window'] = [start, end+time_to_add]
            clip['end_buffer'] += time_to_add
            if i < len(all_clips) - 1:
                all_clips[i+1]['beginning_space'] -= time_to_add
        else:
            pass
        if clip['beginning_buffer'] < 0 and clip['beginning_buffer'] > -0.1: clip['beginning_buffer']=0.
        if clip['end_buffer'] < 0 and clip['end_buffer'] > -0.1: clip['end_buffer']=0.
       
    
    # 4th pass: Compute the class-balance (laughter fraction) for this conversation
    total_window_time = sum([clip['window'][1] - clip['window'][0] for clip in all_clips])
    total_laughter_time = sum([region[1] for region in regions_list])
    swb_laughter_fraction = total_laughter_time / total_window_time
    
    # Tweak this adjustment_amount to find a value for which after everything
    # The class balances match
    intended_window_time = total_laughter_time/(audioset_laughter_fraction) + adjustment_amount

    # 5th pass: Trim back the clips to match the class-balance distribution of the Audioset Annotations
    # Need to reduce the windows to cut 'total_window_time' down to 'intended_window_time'
    # Try to distribute the time so that all windows are close to the same size
    time_to_reduce = total_window_time - intended_window_time
    
    #available_time_per_clip = [clip['beginning_buffer']+clip['end_buffer'] for clip in clips]
    beginning_buffers = [clip['beginning_buffer'] for clip in all_clips]
    end_buffers = [clip['end_buffer'] for clip in all_clips]
    all_buffers = beginning_buffers + end_buffers
    time_to_reduce_per_buffer = distribute_time(time_to_reduce, all_buffers)
    beginning_buffer_updates, end_buffer_updates = np.split(time_to_reduce_per_buffer,2)
    
    
    try:
        for i, clip in enumerate(all_clips):
            assert(clip['beginning_buffer'] >= 0)
            assert(clip['end_buffer'] >= 0)
    except:
        pass
        #import pdb; pdb.set_trace()

    try:
        assert(len(beginning_buffer_updates) == len(all_clips))
        assert(len(end_buffer_updates) == len(all_clips))
    except:
        pass
        #import pdb; pdb.set_trace()
    
    for i, clip in enumerate(all_clips):
        clip['window'][0] += beginning_buffer_updates[i]; clip['beginning_space'] += beginning_buffer_updates[i]
        clip['beginning_buffer'] -= beginning_buffer_updates[i]
        clip['window'][1] -= end_buffer_updates[i]; clip['end_space'] += end_buffer_updates[i]
        clip['end_buffer'] -= end_buffer_updates[i]
        if clip['beginning_buffer'] < 0 and clip['beginning_buffer'] > -0.1: clip['beginning_buffer']=0.
        if clip['end_buffer'] < 0 and clip['end_buffer'] > -0.1: clip['end_buffer']=0.
        
        try:
            assert(clip['beginning_buffer'] >= 0)
            assert(clip['end_buffer'] >= 0)
        except:
            pass
            #import pdb; pdb.set_trace()
        

    # 6th pass: Re-Compute the class-balance (laughter fraction) for this conversation
    total_window_time = sum([clip['window'][1] - clip['window'][0] for clip in all_clips])
    total_laughter_time = sum([region[1] for region in regions_list])
    swb_laughter_fraction = total_laughter_time / total_window_time
    intended_window_time = total_laughter_time/audioset_laughter_fraction
    
    # Now make the dataframe
    rows = []
    # For each window, grab each laughter region that's inside it and mark that relative to the window start
    for i, clip in enumerate(all_clips):
        inside_regions = [r for r in regions_list if audio_utils.times_overlap(
            clip['window'][0], clip['window'][1],r[0],r[0]+r[1])]
        
        if len(inside_regions) > 5:
            pass
            #import pdb; pdb.set_trace()
        
        h = {'FileID': audio_file_path.split('/')[-1].split('.')[0],
                 'audio_path': audio_file_path,
                 'audio_length': full_audio_file_length,
                 'window_start': clip['window'][0],
                 'window_length': clip['window'][1]-clip['window'][0]
                }
        
        for j in range(5):
            if j == 0:
                start_key = 'Start'; end_key = 'End'
            else:
                start_key = f'Start.{j}'; end_key = f'End.{j}'
            if len(inside_regions) > j:
                r = inside_regions[j]
                h[start_key] = r[0]; h[end_key] = r[0] + r[1]
            else:
                h[start_key] = np.nan; h[end_key] = np.nan
        
        if h['window_length'] > 1.:
            rows.append(h)
    
    return rows, all_clips, total_laughter_time, total_window_time


def make_switchboard_dataframe(t_files_A, t_files_B, a_files, adjustment_amount=0,include_words=False):
    all_rows = []; all_clips = []; all_laughter_time = []; all_window_time = []
    all_laughter_regions = []
    for index in tqdm(range(len(a_files))):
        full_audio_file_length = get_audio_file_length(a_files[index])
        laughter_regions, speech_regions, _, _ = dataset_utils.get_laughter_regions_and_speech_regions(
                    t_files_A[index], t_files_B[index], a_files[index], include_words=include_words)
        all_laughter_regions.append(laughter_regions)

        rows, clips, laughter_time, window_time = get_10_second_clips(
            laughter_regions, a_files[index],full_audio_file_length,
            index, audioset_laughter_fraction, adjustment_amount=adjustment_amount)

        all_rows += rows
        all_clips.append(clips)
        all_laughter_time.append(laughter_time)
        all_window_time.append(window_time)
    total_laughter_time = sum(all_laughter_time)
    total_window_time = sum(all_window_time)
    df = pd.DataFrame(all_rows)
    return df

def make_switchboard_distractor_dataframe(t_files_A, t_files_B, a_files, total_distractor_clips=None):
    rows = []
    for i in tqdm(range(len(t_files_A))):
        t_file_A = t_files_A[i]
        t_file_B = t_files_B[i]
        a_file = a_files[i]

        full_audio_file_length = get_audio_file_length(a_file)
        laughter_regions, _, _, _ = dataset_utils.get_laughter_regions_and_speech_regions(
                        t_file_A, t_file_B, a_file, include_words=True)

        laughter_regions = [{'start':r[0], 'end':r[0]+r[1]} for r in laughter_regions]
        non_laughter_regions = get_non_laughter_times(laughter_regions, 0,
            full_audio_file_length, avoid_edges=True, edge_gap=0.5)

        for r in non_laughter_regions:
            if r['end'] - r['start'] > 10:
                start_point = r['start'] + (r['end']-r['start']-10)/2
                end_point = start_point + 10
                h = {'FileID': a_file.split('/')[-1].split('.')[0],
                         'audio_path': a_file,
                         'audio_length': full_audio_file_length,
                         'window_start': start_point,
                         'window_length': 10.
                    }
                for j in range(5):
                    if j == 0:
                        start_key = 'Start'; end_key = 'End'
                    else:
                        start_key = f'Start.{j}'; end_key = f'End.{j}'
                    h[start_key] = np.nan; h[end_key] = np.nan
                rows.append(h)
    if total_distractor_clips is not None:
        rows = rows[0:total_distractor_clips]
    return pd.DataFrame(rows)

swb_val_distractor_df = make_switchboard_distractor_dataframe(
    val_transcription_files_A, val_transcription_files_B, val_audio_files, total_distractor_clips=153)

swb_val_distractor_df.to_csv('../data/switchboard/annotations/clean_switchboard_val_distractor_annotations.csv', index=None)

swb_test_distractor_df = make_switchboard_distractor_dataframe(
    test_transcription_files_A, test_transcription_files_B, test_audio_files, total_distractor_clips=203)

swb_test_distractor_df.to_csv('../data/switchboard/annotations/clean_switchboard_test_distractor_annotations.csv', index=None)
 


swb_train_df = make_switchboard_dataframe(
    train_transcription_files_A, train_transcription_files_B, train_audio_files, adjustment_amount=-1.1)
print("\nSWB Train Annotations stats:")
_, _, _, _, _ = get_annotation_stats(
    swb_train_df, display=True, min_gap = MIN_GAP, avoid_edges=True, edge_gap=0.5)

swb_train_df.to_csv('../data/switchboard/annotations/clean_switchboard_train_laughter_annotations.csv', index=None)

swb_val_df = make_switchboard_dataframe(
    val_transcription_files_A, val_transcription_files_B, val_audio_files, adjustment_amount=-1.1)
print("\nSWB Val Set Annotations stats:")
_, _, _, _, _ = get_annotation_stats(
    swb_val_df, display=True, min_gap = MIN_GAP, avoid_edges=True, edge_gap=0.5)

swb_val_df.to_csv('../data/switchboard/annotations/clean_switchboard_val_laughter_annotations.csv', index=None)


swb_test_df = make_switchboard_dataframe(
    test_transcription_files_A, test_transcription_files_B, test_audio_files, adjustment_amount=1.2)
print("\nSWB Test Set Annotations stats:")
_, _, _, _, _ = get_annotation_stats(
    swb_test_df, display=True, min_gap = MIN_GAP, avoid_edges=True, edge_gap=0.5)

swb_test_df.to_csv('../data/switchboard/annotations/clean_switchboard_test_laughter_annotations.csv', index=None)
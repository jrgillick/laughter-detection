import pandas as pd, numpy as np, os, sys, audioread, librosa
import warnings
import ast
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../utils')
import dataset_utils, audio_utils, data_loaders, torch_utils
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import optim, nn

import laugh_segmenter as laugh_segmenter

def get_audio_file_length(path):
    f = audioread.audio_open(path)
    l = f.duration
    f.close()
    return l

def get_laughter_times_from_annotation_line(line, min_gap=0.0, avoid_edges=True, edge_gap=0.5):
    laughter_segments = []
    if float(line['End']) > 0: laughter_segments.append([float(line['Start']), float(line['End'])])
    for i in range(1,5):
        if not np.isnan(line[f'Start.{i}']): laughter_segments.append([float(line[f'Start.{i}']), float(line[f'End.{i}'])])
    
    window_start = line['window_start']
    window_end = line['window_start'] + line['window_length']

    # Combine annotations if they have less than min_gap seconds between events (because of windowing in the model)
    # Expand time windows to account for minimum gap (window effect)
    laughter_segments = [[np.maximum(0, segment[0]-min_gap),np.minimum(line['audio_length'],segment[1]+min_gap)] for segment in laughter_segments]

    # Merge any overlapping annotations and then convert back to list from tuple
    laughter_segments = dataset_utils.combine_overlapping_regions(laughter_segments, [])
    laughter_segments = [list(s) for s in laughter_segments] 
        
    # Slightly fairer to compare w/ Switchboard if we only take windows for which we see the whole 1 second instead of zero-padding
    # To do this, trim the audio and annotations by 0.5 seconds at start and finish
    trimmed_segments = []
    if avoid_edges:
        for segment in laughter_segments:
            start, end = segment
            # Case when the whole segment is within the edge - skip the segment
            if (start < window_start+edge_gap and end < window_start+edge_gap) or  (start > window_end-edge_gap and end > window_end-edge_gap):
                continue
            # Case when part of the segment is within the edge - modify the segment
            if (start < window_start+edge_gap and end > window_start+edge_gap): 
                segment[0] = window_start+edge_gap
            if (end > window_end-edge_gap and start < window_end-edge_gap):
                try:
                    segment[1] = window_end-edge_gap
                except:
                    import pdb; pdb.set_trace()
            # Otherwise keep the segment unchanged
            trimmed_segments.append(segment)
            
        laughter_segments = trimmed_segments

    # Convert to hash
    laughter_segments = [{'start': segment[0], 'end': segment[1]} for segment in laughter_segments]
    return laughter_segments

# Get all the segments in the audio file that are NOT laughter, using the segments that are laughter and the file length
# Input is array of hashes like [ {'start': 1.107, 'end': 1.858}, {'start': 2.237, 'end': 2.705}]]
def get_non_laughter_times(laughter_segments, window_start, window_length, avoid_edges=True, edge_gap=0.5):
    non_laughter_segments = []
    
    if avoid_edges:
        non_laughter_start=window_start+edge_gap
    else:
        non_laughter_start = window_start
    for segment in laughter_segments:
        non_laughter_end = segment['start']
        if non_laughter_end > non_laughter_start:
            non_laughter_segments.append({'start': non_laughter_start, 'end': non_laughter_end})
        non_laughter_start = segment['end']
    
    if avoid_edges:
        non_laughter_end= window_start + window_length - edge_gap
    else:
        non_laughter_end = window_length
    
    if non_laughter_end > non_laughter_start:
        non_laughter_segments.append({'start': non_laughter_start, 'end': non_laughter_end})
    return non_laughter_segments

def sum_overlap_amount(true_segments, predicted_segments):
    total = 0.
    for ts in true_segments:
        for ps in predicted_segments:
            total += overlap_amount(ts['start'], ts['end'], ps['start'], ps['end'])
    return total

def times_overlap(start1, end1, start2, end2):
    if end1 <= start2 or end2 <= start1:
        return False
    else:
        return True

def overlap_amount(start1, end1, start2, end2):
    if not times_overlap(start1, end1, start2, end2):
        return 0.
    # Equal on one side
    elif start1 == start2:
        return np.minimum(end1, end2) - start1
    elif end1 == end2:
        return end1 - np.maximum(start1, start2)
    # One contained totally within the other
    elif start2 > start1 and end2 < end1:
        return end2-start2
    elif start1 > start2 and end1 < end2:
        return end1-start1
    # Overlap on one side
    elif end1 > start2 and start1 < start2:
        return end1 - start2
    elif end2 > start1 and start2 < start1:
        return end2 - start1

def get_annotation_stats(annotations_df, display=True, min_gap=0.0, avoid_edges=True, edge_gap=0.5):
    laughter_lengths = []
    non_laughter_lengths = []
    total_lengths = []
    laughter_count = 0
    
    for i in range(len(annotations_df)):
        
        line = dict(annotations_df.iloc[i])
        
        #audio_length = annotations_df.iloc[i].audio_length
        times = get_laughter_times_from_annotation_line(
            line,min_gap=min_gap,avoid_edges=avoid_edges,edge_gap=edge_gap)
        laughter_count += len(times)
        
        no_times = get_non_laughter_times(
            times, line['window_start'], line['window_length'], avoid_edges=avoid_edges, edge_gap=edge_gap)

        #laughter_segments, window_start, window_length, avoid_edges=False, edge_gap=0.5
        
        laughter_length = sum_overlap_amount(times, times)
        non_laughter_length = sum_overlap_amount(no_times, no_times)
        
        total_length = laughter_length + non_laughter_length
        #non_laughter_length = total_length - laughter_length
        laughter_lengths.append(laughter_length)
        non_laughter_lengths.append(non_laughter_length)
        total_lengths.append(total_length)
        
    total_minutes = np.sum(total_lengths)/60
    total_laughter_minutes = np.sum(laughter_lengths)/60
    total_non_laughter_minutes = np.sum(non_laughter_lengths)/60
    laughter_fraction =total_laughter_minutes/total_minutes

    if display:
        print(f"Total minutes in annotations: {total_minutes}")
        print(f"Total laughter minutes in annotations: {total_laughter_minutes}")
        print(f"Total non-laughter minutes in annotations: {total_non_laughter_minutes}")
        print(f"Percentage of laughter in annotations: {laughter_fraction}")
        print(f"Number of distinct laughs identified: {laughter_count}")
    
    return total_minutes, total_laughter_minutes, total_non_laughter_minutes, laughter_fraction, laughter_count

def predict_laughter_times(model, line, config, model_input_size=1.,
                          use_filter=False, threshold=0.5,min_length=0.,
                          avoid_edges = True, edge_gap=0.5, expand_channel_dim=False):

    audio_path = line['audio_path']
    offset = line['window_start']
    duration = line['window_length']
    feature_fn = config['feature_fn']

    if avoid_edges:
        in_window_duration = duration - 2*edge_gap
        y, sr = librosa.load(audio_path, sr=8000, offset=offset, duration=duration)
    else:
        # Extend if possible
        if offset > model_input_size/2 and offset + duration + model_input_size/2 < line['audio_length']:
            offset -= model_input_size/2
            duration += model_input_size
            y, sr = librosa.load(audio_path, sr=8000, offset=offset, duration=duration)
        # Else pad w/ zeros
        else:
            y, sr = librosa.load(audio_path, sr=8000, offset=offset, duration=duration)
            z = np.zeros(int(sr*(model_input_size/2)))
            y = np.concatenate([z,y,z])

    feats = np.ascontiguousarray(feature_fn(y=y, sr=sr, offset=None, duration=None))
    windowed_feats = librosa.util.frame(feats,frame_length=44,hop_length=1,axis=0)
    #windowed_feats = windowed_feats.reshape((len(windowed_feats),config['linear_layer_size']))
    if expand_channel_dim:
        windowed_feats = np.expand_dims(windowed_feats, 1)
    model.eval()
    inputs = torch.from_numpy(windowed_feats).float().to(device)
    
    all_probs = []
    batch_size = 8
    for k in range(0, len(inputs), batch_size):
        batch = inputs[k:k+batch_size]
        if len(batch)>0:
            probs = model(batch).detach().cpu().numpy()
            if len(probs) == 1:
                all_probs.append(probs[0][0])
            else:
                all_probs += list(probs.squeeze())
    probs = np.array(all_probs)    
    
    fps = len(probs)/in_window_duration

    if use_filter:
        probs = laugh_segmenter.lowpass(probs)

    predicted_laughter_times = laugh_segmenter.get_laughter_instances(
        probs, threshold=threshold, min_length=min_length, fps=fps)
    
    if avoid_edges:
        predicted_laughter_times = [(inst[0]+edge_gap,inst[1]+edge_gap) for inst in predicted_laughter_times]
        
    if line['window_start']>0:
        predicted_laughter_times = [(inst[0]+line['window_start'],inst[1]+line['window_start']) for inst in predicted_laughter_times]
    
    predicted_laughter_times = [{'start':segment[0],'end':segment[1]} for segment in predicted_laughter_times]
    
    return predicted_laughter_times

def get_results_for_annotation_index(model, config, annotations_df, index, min_gap=0.,
                                     threshold=0.5, use_filter=False, min_length=0.0,
                                     avoid_edges=True, edge_gap=0.5, expand_channel_dim=False):
    
    line = dict(annotations_df.iloc[index])
    audio_path = line['audio_path']    
    
    true_laughter_times = get_laughter_times_from_annotation_line(
            line,min_gap=min_gap,avoid_edges=avoid_edges,edge_gap=edge_gap)
    true_non_laughter_times = get_non_laughter_times(
            true_laughter_times,line['window_start'],line['window_length'],avoid_edges=avoid_edges,edge_gap=edge_gap)
    
    predicted_laughter_times = predict_laughter_times(
        model, line, config, model_input_size=1.,
        use_filter=use_filter, threshold=threshold,min_length=min_length,
        avoid_edges=avoid_edges, edge_gap=edge_gap, expand_channel_dim=expand_channel_dim)

    predicted_non_laughter_times = get_non_laughter_times(
        predicted_laughter_times,line['window_start'],line['window_length'],avoid_edges=avoid_edges,edge_gap=edge_gap)
    
    total_laughter_time = sum_overlap_amount(true_laughter_times,true_laughter_times)
    total_non_laughter_time = sum_overlap_amount(true_non_laughter_times,true_non_laughter_times)

    true_positive_time = sum_overlap_amount(true_laughter_times, predicted_laughter_times)
    true_negative_time = sum_overlap_amount(true_non_laughter_times, predicted_non_laughter_times)
    false_positive_time = sum_overlap_amount(true_non_laughter_times, predicted_laughter_times)
    false_negative_time = sum_overlap_amount(true_laughter_times, predicted_non_laughter_times)
    
    total_time = true_positive_time + true_negative_time + false_positive_time + false_negative_time
    
    #import pdb; pdb.set_trace()
    
    try:
        assert(np.abs(total_laughter_time - (true_positive_time + false_negative_time)) < 0.2)
        assert(np.abs(total_non_laughter_time - (true_negative_time + false_positive_time)) < 0.2)
    except:
        print(index)
        print(line['window_length'])
        print(np.abs(total_laughter_time - (true_positive_time + false_negative_time)))
        print("\n") 
    
    h = {'FileID':annotations_df.FileID[index], 'tp_time':true_positive_time, 'tn_time':true_negative_time,
         'fp_time':false_positive_time, 'fn_time':false_negative_time,
         'predicted_laughter': predicted_laughter_times, 'predicted_non_laughter': predicted_non_laughter_times,
         'true_laughter': true_laughter_times, 'true_non_laughter': true_non_laughter_times}

    return h


# Methods for calculating event metrics

def get_results_row(df, index):
    true_laughter = ast.literal_eval(df.true_laughter[index])
    predicted_laughter = ast.literal_eval(df.predicted_laughter[index])
    true_non_laughter = ast.literal_eval(df.true_non_laughter[index])
    predicted_non_laughter = ast.literal_eval(df.predicted_non_laughter[index])
    return true_laughter, predicted_laughter, true_non_laughter, predicted_non_laughter

def is_inside_window(window_a, window_b):
    # windows are dicts with keys 'start' and 'end'
    # returns True if window_a is completed inside window_b
    return (window_a['start']>= window_b['start'] and window_a['end']<=window_b['end'])

def overlap_length(window_a, window_b):
    return overlap_amount(window_a['start'], window_a['end'],
                                     window_b['start'], window_b['end'])

def is_outside_all_windows(window_a, all_windows):
    for other_window in all_windows:
        if overlap_length(window_a, other_window) > 0:
            return False
    return True

def window_length(window):
    return window['end'] - window['start']
    
def get_event_metrics_per_row(df, index, cutoff_length=0.2):
    # convert to metrics for F1 on events
    # True positives:
        # For each true laugh, if there is a prediction during that time that lasts >0.1s, count it. (not more than 1)
            # True annotated events lasting less than 0.1 secs are IGNORED for this
    # False positives:
        # For each prediction, if > 0.1s is completely outside the true positives, count it.
    # True negatives:
        # For each true negative region, if there is no prediction overlapping the window by > 0.1s, count it
    # False negatives:
            # True annotated events lasting less than 0.1 secs are IGNORED for this
    # For each true laugh, if there is no prediction completely inside the window lasting >0.2s, count it.
    # inputs:
    #  df - Pandas dataframe with prediction results
    #  index - row index into the dataframe
    #  cutoff_length - minimum length for a time window to be counted as an "event"
    # Returns:
    # number of true positives, false positives, true negatives, false negatives
    # tp, fp, tn, fn
    t_laughs, p_laughs, t_non_laughs, p_non_laughs = get_results_row(df,index)
    tp = 0; fp = 0; tn = 0; fn = 0
    # Count TP and FN
    for t_laugh in t_laughs:
        if window_length(t_laugh) > cutoff_length:
            pred_count = 0
            for p_laugh in p_laughs:
                if overlap_length(p_laugh, t_laugh) >= cutoff_length:
                    pred_count += 1
            if pred_count == 0:
                fn += 1
            else:
                tp += 1
        
    # Count FP
    for p_laugh in p_laughs:
        if is_outside_all_windows(p_laugh, t_laughs) and window_length(p_laugh) > cutoff_length:
            fp += 1
        #for t_non_laugh in t_non_laughs:
        #    if overlap_length(p_laugh, t_non_laugh) > cutoff_length:
        #        fp += 1
        #        break
            
    # Count TN
    #for t_non_laugh in t_non_laughs:
    #    if window_length(t_non_laugh) > cutoff_length:
    #        preds = [overlap_length(p_laugh, t_non_laugh) > cutoff_length for p_laugh in p_laughs]
    #        #preds = [is_inside_window(p_laugh, t_non_laugh) and window_length(p_laugh) > cutoff_length for p_laugh in p_laughs]
    #        preds = [p for p in preds if p]
    #        if len(preds) == 0:
    #            tn += 1

    # Don't worry about defining true negatives here since not needed for F-score
    return tp, fp, 0, fn
        


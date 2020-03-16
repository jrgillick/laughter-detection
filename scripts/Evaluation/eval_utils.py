import pandas as pd, numpy as np, os, sys, audioread
import warnings
sys.path.append('/mnt/data0/jrgillick/projects/audio-feature-learning/')
import dataset_utils, audio_utils, data_loaders

def get_audio_file_length(path):
    f = audioread.audio_open(path)
    l = f.duration
    f.close()
    return l

def get_laughter_times_from_annotation_line(line, min_gap=0.5, avoid_edges=False, edge_gap=0.5):
    laughter_segments = []
    if float(line['End']) > 0: laughter_segments.append([float(line['Start']), float(line['End'])])
    for i in range(1,5):
        if not np.isnan(line[f'Start.{i}']): laughter_segments.append([float(line[f'Start.{i}']), float(line[f'End.{i}'])])

    extra_beginning_time = line['extra_beginning_time'] if 'extra_beginning_time' in line.keys() else None
    extra_end_time = line['extra_end_time'] if 'extra_end_time' in line.keys() else None
        
    # Audioset case
    if extra_beginning_time is None and extra_end_time is None:
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
                if (start < edge_gap and end < edge_gap) or  (start > line['audio_length']-edge_gap and end > line['audio_length']-edge_gap):
                    continue
                # Case when part of the segment is within the edge - modify the segment
                if (start < edge_gap and end > edge_gap): 
                    segment[0] = edge_gap
                if (end > line['audio_length']-edge_gap and start < line['audio_length']-edge_gap):
                    try:
                        segment[1] = line['audio_length']-edge_gap
                    except:
                        import pdb; pdb.set_trace()
                # Otherwise keep the segment unchanged
                trimmed_segments.append(segment)
            
            laughter_segments = trimmed_segments

        # Convert to hash
        laughter_segments = [{'start': segment[0], 'end': segment[1]} for segment in laughter_segments]
        return laughter_segments
    
    # Switchboard case
    else:
        # In this case with SWB, we only should have 1 segment per line in the Data frame and it's window has been expanded already
        assert(len(laughter_segments)==1)
        segment = laughter_segments[0]
        segment_length = segment[1]-segment[0]
        segment[1] = segment_length + extra_beginning_time
        segment[0] = extra_beginning_time
        laughter_segments = [{'start': segment[0], 'end': segment[1]}]
        return laughter_segments

# Get all the segments in the audio file that are NOT laughter, using the segments that are laughter and the file length
# Input is array of hashes like [ {'start': 1.107, 'end': 1.858}, {'start': 2.237, 'end': 2.705}]]
def get_non_laughter_times(laughter_segments, file_length, avoid_edges=False, edge_gap=0.5):
    non_laughter_segments = []
    
    if avoid_edges:
        non_laughter_start=edge_gap
    else:
        non_laughter_start = 0.0
    for segment in laughter_segments:
        non_laughter_end = segment['start']
        if non_laughter_end > non_laughter_start:
            non_laughter_segments.append({'start': non_laughter_start, 'end': non_laughter_end})
        non_laughter_start = segment['end']
    
    if avoid_edges:
        non_laughter_end=file_length-edge_gap
    else:
        non_laughter_end = file_length
    
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
    
def get_baseline_results_per_annotation_index(model, annotations_df,
                                              baseline_laugh_segmenter, i, min_gap=0.375,
                                              threshold=0.5, use_filter=True, min_length=0.1,
                                              avoid_edges=False, edge_gap=0.5):
    audio_file = annotations_df.audio_path.iloc[i]
    
    extra_beginning_time = annotations_df.iloc[i].extra_beginning_time if 'extra_beginning_time' in list(annotations_df.columns) else None
    extra_end_time = annotations_df.iloc[i].extra_end_time if 'extra_end_time' in list(annotations_df.columns) else None
    line = dict(annotations_df.iloc[i])
    
    # Switchboard
    if extra_beginning_time is not None and extra_end_time is not None:
        true_laughter_times = get_laughter_times_from_annotation_line(line, min_gap=min_gap)
        absolute_start_time = line['Start'] - extra_beginning_time
        absolute_end_time = line['End'] + extra_end_time
        audio_length = absolute_end_time - absolute_start_time
        true_non_laughter_times = get_non_laughter_times(true_laughter_times, audio_length)
        predicted_laughter_times = baseline_laugh_segmenter.segment_laugh_with_model(
            model, input_path=audio_file, threshold=threshold, use_filter=use_filter, min_length=min_length,
            audio_start=absolute_start_time, audio_length=audio_length)
        predicted_non_laughter_times = get_non_laughter_times(predicted_laughter_times, audio_length)
    # Audioset
    else:
        audio_length = annotations_df.iloc[i].audio_length
        true_laughter_times = get_laughter_times_from_annotation_line(line, min_gap=min_gap, avoid_edges=avoid_edges)
        true_non_laughter_times = get_non_laughter_times(true_laughter_times, audio_length, avoid_edges=avoid_edges)
        predicted_laughter_times = baseline_laugh_segmenter.segment_laugh_with_model(
            model, input_path=audio_file, threshold=threshold, use_filter=use_filter, min_length=min_length,
            avoid_edges=False, edge_gap=edge_gap)
        predicted_non_laughter_times = get_non_laughter_times(predicted_laughter_times, audio_length, avoid_edges=avoid_edges)

    total_laughter_time = sum_overlap_amount(true_laughter_times,true_laughter_times)
    total_non_laughter_time = sum_overlap_amount(true_non_laughter_times,true_non_laughter_times)

    true_positive_time = sum_overlap_amount(true_laughter_times, predicted_laughter_times)
    true_negative_time = sum_overlap_amount(true_non_laughter_times, predicted_non_laughter_times)
    false_positive_time = sum_overlap_amount(true_non_laughter_times, predicted_laughter_times)
    false_negative_time = sum_overlap_amount(true_laughter_times, predicted_non_laughter_times)
    
    total_time = true_positive_time + true_negative_time + false_positive_time + false_negative_time
    
    try:
        assert(np.abs(total_laughter_time - (true_positive_time + false_negative_time)) < 0.1)
        assert(np.abs(total_non_laughter_time - (true_negative_time + false_positive_time)) < 0.1)
        if avoid_edges:
            assert(np.abs(total_time - (audio_length - 2*edge_gap)) < 0.1)
        else:
            assert(np.abs(total_time - audio_length) < 0.1)
        assert(np.abs(total_time - (total_laughter_time + total_non_laughter_time)) < 0.1)
    except:
        import pdb; pdb.set_trace()
        
    h = {'FileID':annotations_df.FileID[i], 'tp_time':true_positive_time, 'tn_time':true_negative_time,
         'fp_time':false_positive_time, 'fn_time':false_negative_time,
         'predicted_laughter': predicted_laughter_times, 'predicted_non_laughter': predicted_non_laughter_times,
         'true_laughter': true_laughter_times, 'true_non_laughter': true_non_laughter_times}

    return h

def get_annotation_stats(annotations_df, display=True, min_gap=0.5, avoid_edges=False, edge_gap=0.5):
    laughter_lengths = []
    non_laughter_lengths = []
    total_lengths = []
    laughter_count = 0
    
    for i in range(len(annotations_df)):
        extra_beginning_time = annotations_df.iloc[i].extra_beginning_time if 'extra_beginning_time' in list(annotations_df.columns) else None
        extra_end_time = annotations_df.iloc[i].extra_end_time if 'extra_end_time' in list(annotations_df.columns) else None
        line = dict(annotations_df.iloc[i])
        
        #Switchboard
        if extra_beginning_time is not None and extra_end_time is not None:
            times = get_laughter_times_from_annotation_line(line, min_gap=min_gap)
            laughter_count += len(times)
            laughter_length = sum_overlap_amount(times, times)
            absolute_start_time = line['Start'] - extra_beginning_time
            absolute_end_time = line['End'] + extra_end_time
            total_length = absolute_end_time - absolute_start_time
            non_laughter_times = get_non_laughter_times(times, total_length)
            non_laughter_length = total_length - laughter_length
            laughter_lengths.append(laughter_length)
            non_laughter_lengths.append(non_laughter_length)
            total_lengths.append(total_length)
        #Audioset
        else:
            audio_length = annotations_df.iloc[i].audio_length
            times = get_laughter_times_from_annotation_line(line, min_gap=min_gap,avoid_edges=avoid_edges)
            laughter_count += len(times)
            
            if avoid_edges: 
                total_length = annotations_df.iloc[i].audio_length - 2*edge_gap
            else:
                total_length = annotations_df.iloc[i].audio_length
            
            laughter_length = sum_overlap_amount(times, times)
            non_laughter_length = total_length - laughter_length
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
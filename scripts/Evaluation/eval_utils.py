import pandas as pd
import numpy as np, os, sys
import warnings

def get_laughter_times_from_annotation_line(h):
    laughter_segments = []
    if float(h['Start']) > 0:
        laughter_segments.append({'start': float(h['Start']), 'end': float(h['End'])})
    for i in range(1,5):
        if not np.isnan(h[f'Start.{i}']):
            laughter_segments.append({'start': float(h[f'Start.{i}']), 'end': float(h[f'End.{i}'])})
    return laughter_segments

# Get all the segments in the audio file that are NOT laughter, using the segments that are laughter and the file length
# Input is array of hashes like [ {'start': 1.107, 'end': 1.858}, {'start': 2.237, 'end': 2.705}]]
def get_non_laughter_times(laughter_segments, file_length):
    non_laughter_segments = []
    
    non_laughter_start = 0.0
    for segment in laughter_segments:
        non_laughter_end = segment['start']
        if non_laughter_end > non_laughter_start:
            non_laughter_segments.append({'start': non_laughter_start, 'end': non_laughter_end})
        non_laughter_start = segment['end']
        
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
    if end1 < start2 or end2 < start1:
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
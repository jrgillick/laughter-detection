import pandas as pd, numpy as np, math, os
from praatio import tgio
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed
import text_utils
import audio_utils
import itertools
from tqdm import tqdm

"""
# Dataset or Format Specific Utils
"""

##################### SWITCHBOARD (Laughter Detection)  #############################

# methods for getting files from Switchboard Corpus
def get_train_val_test_folders(t_root):
    t_folders = [t_root + f for f in os.listdir(t_root) if os.path.isdir(t_root + f)]
    t_folders.sort()
    train_folders = t_folders[0:23]
    val_folders = t_folders[23:26]
    test_folders = t_folders[26:30]
    train_folders.sort(); val_folders.sort(); test_folders.sort()
    return (train_folders, val_folders, test_folders)

def get_transcription_files(folder):
    return [f for f in librosa.util.find_files(folder,ext='text') if f.endswith('word.text')]

def get_laughter_rows_from_file(f, include_words=False):
    if include_words:
        return [l for l in get_text_from_file(f) if 'laughter' in l]
    else:
        return [l for l in get_text_from_file(f) if '[laughter]' in l] # doesn't allow laughter with words together

def get_audio_file_from_id(d, all_audio_files):
    files = [f for f in all_audio_files if d in f]
    if len(files) == 1:
        return files[0]
    elif len(files) > 1:
        print("Warning: More than 1 audio file matched id %d" % (int(d)))
        return None
    else:
        print("Warning: No audio file matched id %d" % (int(d)))
        return None

def get_id_from_row(row):
    return row[2:6]

def get_id_from_file(f):
    return get_id_from_row(get_text_from_file(f)[0])

def get_audio_file_from_row(row, all_audio_files):
    return get_audio_file_from_id(get_id_from_row(row))

def get_audio_file_from_transcription_text(t, all_audio_files):
    return get_audio_file_from_id(get_id_from_row(t[0]))

def get_audio_file_from_transcription_file(f, all_audio_files):
    t = open(f).read().split('\n')
    return get_audio_file_from_id(get_id_from_row(t[0]), all_audio_files)

def extract_times_from_row(row):
    return (float(row.split()[1]), float(row.split()[2]))

def get_length_from_transcription_file(t_file):
    try:
        return float(open(t_file).read().split('\n')[-2].split()[-2])
    except:
        print(t_file)

# a_or_b should be either 'A' or 'B' - referring to label of which speaker
def get_transcriptions_files(folder, a_or_b):
    files = []
    subfolders = [folder + "/" + f for f in os.listdir(folder)]
    for f in subfolders:
        fs = [f + "/" + fname for fname in os.listdir(f) if 'a-word.text' in fname and a_or_b in fname]
        files += fs
    files.sort()
    return files

def get_all_transcriptions_files(folder_list, a_or_b):
    files = []
    for folder in folder_list:
        files += get_transcriptions_files(folder, a_or_b)
    files.sort()
    return files

def get_audio_files_from_transcription_files(transcription_files, all_audio_files):
    files = []
    transcription_files_to_remove = []
    for f in transcription_files:
        audio_file = get_audio_file_from_transcription_file(f, all_audio_files)
        if audio_file is None:
            transcription_files_to_remove.append(f)
        else:
            files.append(audio_file)
    #files = list(set(files))
    transcription_files = [t for t in transcription_files if t not in transcription_files_to_remove]
    return transcription_files, files

# Check if laughter is present in a region of an audio file by looking at the transcription file
def no_laughter_present(t_files,start,end):
    for t_file in t_files:
        all_rows = get_text_from_file(t_file)
        for row in all_rows:
            region_start, region_end = extract_times_from_row(row)
            if audio_utils.times_overlap(float(region_start), float(region_end), float(start), float(end)):
                if 'laughter' in row.split()[-1]:
                    return False
    return True

def get_random_speech_region_from_files(t_files, audio_length, region_length, random_seed=None):
    contains_laughter = True
    tries = 0
    while(contains_laughter):
        tries += 1
        if tries > 10:
            print("audio length %f" % (audio_length))
            print("region length %f" % (region_length))
            return None
        if random_seed is not None:
            np.random.seed(random_seed+tries)
        start = np.random.uniform(1.0, audio_length - region_length - 1.0)
        end = start + region_length
        if no_laughter_present(t_files,start,end):
            contains_laughter = False
    return (start, end)

def get_laughter_regions_from_file(t_file, include_words=False):
    rows = get_laughter_rows_from_file(t_file,include_words=include_words)
    times = []
    for row in rows:
        try:
            start, end = extract_times_from_row(row)
            if end - start > 0.05:
                times.append((start,end))
        except:
            continue
    return times

def get_text_from_file(f):
    return (open(f).read().split("\n"))[0:-1]

def combine_overlapping_regions(regions_A, regions_B):
    all_regions = regions_A + regions_B
    overlap_found = True
    while(overlap_found):
        i = 0; j = 0
        overlap_found = False
        while i < len(all_regions):
            j = 0
            while j < len(all_regions):
                if i < j:
                    start1 = all_regions[i][0]; end1 = all_regions[i][1]
                    start2 = all_regions[j][0]; end2 = all_regions[j][1]
                    if audio_utils.times_overlap(start1, end1, start2, end2):
                        overlap_found = True
                        all_regions.pop(i); all_regions.pop(j-1)
                        all_regions.append((min(start1, start2), max(end1, end2)))
                j += 1
            i += 1
    return sorted(all_regions, key=lambda r: r[0])

def get_laughter_regions_and_speech_regions(text_A, text_B, audio_file, random_seed=None, include_words=False):
    # For switchboard laughter. Given a list of files in a partition (train,val, or test)
    # extract all the start and end times for laughs, and sample an equal number of negative examples.
    # When making the text dataset, store columns indicating the full start and end times of an event.
    # For example, start at 6.2 seconds and end at 12.9 seconds
    # We store another column with subsampled start and end times (1 per event)
    # and a column with the length of the subsample (typically always 1.0).
    # Then the data loader can have an option to do subsampling every time (e.g. during training)
    # or to use the pre-sampled times (e.g. during validation)
    # If we want to resample the negative examples (since there are more negatives than positives)
    # then we need to call this function again.
    laughter_regions_A = get_laughter_regions_from_file(text_A,include_words=include_words)
    laughter_regions_B = get_laughter_regions_from_file(text_B,include_words=include_words)
    laughter_regions = combine_overlapping_regions(laughter_regions_A, laughter_regions_B)
    speech_regions = []
    audio_length = get_length_from_transcription_file(text_A)
    for laughter_region in laughter_regions:
        region_length = laughter_region[1] - laughter_region[0]
        speech_regions.append(get_random_speech_region_from_files([text_A, text_B], audio_length, region_length, random_seed=random_seed))
    laughter_regions = [l for l in laughter_regions if l is not None]
    speech_regions = [s for s in speech_regions if s is not None]
    laughter_regions = [audio_utils.start_end_to_offset_duration(s,e) for s,e in laughter_regions]
    speech_regions = [audio_utils.start_end_to_offset_duration(s,e) for s,e in speech_regions]
    # Add padding on each side for windowing later
    laughter_region_subsamples = [audio_utils.subsample_time(
        l[0], l[1], audio_length, subsample_length=1.0, padding_length=0.5, random_seed=random_seed) for l in laughter_regions]
    speech_region_subsamples = [audio_utils.subsample_time(
        s[0], s[1], audio_length, subsample_length=1.0, padding_length=0.5, random_seed=random_seed) for s in speech_regions]
    return laughter_regions, speech_regions, laughter_region_subsamples, speech_region_subsamples

def get_laughter_speech_text_lines(t_file_a, t_file_b, a_file, convert_to_text=True, random_seed=None, include_words=False):
    # Columns: [region start, region duration, subsampled region start, subsampled region duration, audio path, label]
    lines = []
    laughter_regions, speech_regions, laughter_region_subsamples, speech_region_subsamples = get_laughter_regions_and_speech_regions(
        t_file_a,t_file_b,a_file,random_seed=random_seed, include_words=include_words)
    assert(len(laughter_regions) == len(laughter_region_subsamples))
    assert(len(speech_regions) == len(speech_region_subsamples))
    for i in range(len(laughter_regions)):
        r = laughter_regions[i]
        r_subsample = laughter_region_subsamples[i]
        # Columns: [region start, region duration, subsampled region start, subsampled region duration, audio path, label]
        line = list(r) + list(r_subsample) + [a_file] + [1]
        if convert_to_text:
            lines.append('\t'.join([str(l) for l in line]))
        else:
            lines.append(line)
    for i in range(len(speech_regions)):
        r = speech_regions[i]
        r_subsample = speech_region_subsamples[i]
        line = list(r) + list(r_subsample) + [a_file] + [0]
        if convert_to_text:
            lines.append('\t'.join([str(l) for l in line]))
        else:
            lines.append(line)
    return lines

def sample_switchboard_features_and_labels(audio_file, t_file_a, t_file_b, feature_fn, random_seed=None, include_words=False, **kwargs):
    laughter_regions, speech_regions = get_laughter_regions_and_speech_regions(
        t_file_a, t_file_b, audio_file, random_seed=random_seed, include_words=include_words)
    laughter_feats = audio_utils.featurize_audio_segments(laughter_regions, feature_fn=feature_fn,f=audio_file, sr=8000)
    speech_feats = audio_utils.featurize_audio_segments(speech_regions, feature_fn=feature_fn,f=audio_file, sr=8000)
    X = laughter_feats+speech_feats
    y = list(np.ones(len(laughter_regions))) + list(np.zeros(len(speech_regions)))
    return list(zip(X,y))




import numpy as np
import librosa
import os
import sys
import audioread
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import pickle

# methods for getting files from Switchboard Corpus
def get_train_val_test_folders(t_root):
    t_folders = [t_root + f for f in os.listdir(t_root) if os.path.isdir(t_root + f)]
    t_folders.sort()
    train_folders = t_folders[0:20]
    val_folders = t_folders[20:25]
    test_folders = t_folders[25:30]
    train_folders.sort(); val_folders.sort(); test_folders.sort()
    return (train_folders, val_folders, test_folders)

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

def get_transcription_files_with_laughter_in_corpus(folder_list, a_or_b):
    files = []
    transcription_files = get_all_transcriptions_files(folder_list, a_or_b)
    for f in transcription_files:
        if count_laughter_instances_in_transcription_file(f) > 0:
            files.append(f)
    return files

def count_transcription_files_with_laughter_in_corpus(folder_list, a_or_b):
    return len(get_transcription_files_with_laughter_in_corpus(folder_list, a_or_b))

def get_sph_files(folder):
    return [folder + "/" + f for f in os.listdir(folder) if ".sph" in f]

def get_all_audio_files(a_root):
    files = []
    a_folders = [a_root + f + "/data" for f in os.listdir(a_root) if os.path.isdir(a_root + f)]
    a_folders.sort()
    for folder in a_folders:
        files += get_sph_files(folder)
    files.sort()
    return files

def get_text_from_file(f):
    return (open(f).read().split("\n"))[0:-1]

def get_laughter_rows_from_file(f):
    #return [l for l in get_text_from_file(f) if 'laughter' in l]
    return [l for l in get_text_from_file(f) if '[laughter]' in l] # doesn't allow laughter with words together

def get_audio_file_from_id(d):
    files = [f for f in all_audio_files if d in f]
    if len(files) == 1:
        return files[0]
    elif len(files) > 1:
        print "Warning: More than 1 audio file matched id %d" % (int(d))
        return None
    else:
        print "Warning: No audio file matched id %d" % (int(d))
        return None
        
def get_id_from_row(row):
    return row[2:6]

def get_id_from_file(f):
    return get_id_from_row(get_text_from_file(f)[0])

def get_audio_file_from_row(row):
    return get_audio_file_from_id(get_id_from_row(row))

def get_audio_file_from_transcription_text(t):
    return get_audio_file_from_id(get_id_from_row(t[0]))

def get_audio_file_from_transcription_file(f):
    t = open(f).read().split('\n')
    return get_audio_file_from_id(get_id_from_row(t[0]))

def get_audio_file_length(path):
    f = audioread.audio_open(path)
    l = f.duration
    f.close()
    return l

def count_laughter_instances_in_transcription_file(f):
    rows = get_laughter_rows_from_file(f)
    return len(rows)

def count_laughter_instances_in_corpus(folder_list, a_or_b):
    transcription_files = get_all_transcriptions_files(folder_list, a_or_b)
    count = 0
    for f in transcription_files:
        count += count_laughter_instances_in_transcription_file(f)
    return count

def get_audio_files_from_transcription_files(transcription_files):
    files = []
    for f in transcription_files:
        files.append(get_audio_file_from_transcription_file(f))
    files = list(set(files))
    files.sort()
    if None in files: files.remove(None)
    return files

def extract_times_from_row(row):
    return (float(row.split(' ')[1]), float(row.split(' ')[2]))

def get_laughter_regions_from_file(t_file):
    rows = get_laughter_rows_from_file(t_file)
    times = []
    for row in rows:
        start, end = extract_times_from_row(row)
        if end - start > 0.05:
            times.append((start,end))
    return times

def get_length_from_regions_list(times):
    return sum([end - start for start, end in times])
    
def get_random_speech_region_from_file(t_file, region_length):
    audio_length = get_audio_file_length(get_audio_file_from_transcription_file(t_file))
    contains_laughter = True
    tries = 0
    while(contains_laughter):
        tries += 1
        if tries > 10:
            print "audio length %f" % (audio_length)
            print "region legnth %f" % (region_length)
            return None
        start = np.random.uniform(1.0, audio_length - region_length - 1.0)
        end = start + region_length
        if no_laughter_present(t_file,start,end):
            contains_laughter = False
    return (start, end)

# Check if laughter is present in a region of an audio file by looking at the transcription file
def no_laughter_present(t_file,start,end):
    all_rows = get_text_from_file(t_file)
    for row in all_rows:
        region_start, region_end = extract_times_from_row(row)
        if times_overlap(float(region_start), float(region_end), float(start), float(end)):
            if 'laughter' in row.split(' ')[-1]:
                return False
    return True
        
def times_overlap(start1, end1, start2, end2):
    if end1 < start2 or end2 < start1:
        return False
    else:
        return True





#Methods for processing audio and computing MFCC and Delta features

# pad with 0.5 seconds on each side of the desired region
def clip_audio_region(y,sr,start,end,pad_amount=0.5):
    start_sample = int((start-pad_amount)*sr)
    end_sample = int((end+pad_amount)*sr)
    return y[start_sample:end_sample]

def write_clip_to_disk(path,y,sr):
    librosa.output.write_wav(path,y,sr)
    
def compute_mfcc_features(y,sr):
    return mfcc(y,samplerate=sr,winlen=0.025,winstep=0.01)

def compute_delta_features(mfcc_feat):
    return delta(mfcc_feat, 2)

def compute_labels_per_frame(n_frames,sr,winstep=0.01,pad_amount=0.5):
    #print "n_frames: %d" % (n_frames)
    samples_per_frame = sr*winstep #80 with defaults
    #with 0.5 seconds of padding, there should be 4000 samples of padding, so 50 frames of non-laughter 
    n_padding_frames = int(sr * pad_amount / samples_per_frame)
    padding_frames = list(np.zeros(n_padding_frames))
    laughter_frames = list(np.ones(n_frames - 2*n_padding_frames))
    labels = padding_frames + laughter_frames + padding_frames
    return labels

def compute_features_and_labels(y,sr,region,label_type,source_file_id,file_index):
    clip = clip_audio_region(y,sr,start=region[0],end=region[1])
    mfcc_features = compute_mfcc_features(clip,sr)
    delta_features = compute_delta_features(mfcc_features)
    n_frames = len(mfcc_features)
    if label_type == 'laughter':
        labels = compute_labels_per_frame(n_frames,sr)
    else:
        labels = np.zeros(n_frames)
    return {'mfcc': mfcc_features,
            'delta': delta_features,
            'labels': labels,
            'clip_type': label_type,
            'source_file_id': source_file_id,
            'file_index': file_index}

def compute_and_store_features_and_labels(t_file, output_dir, a_or_b):
    a_file = get_audio_file_from_transcription_file(t_file)
    y,sr = librosa.load(a_file,sr=8000)
    source_file_id = get_id_from_file(t_file)
    laughter_regions = get_laughter_regions_from_file(t_file)
    
    laughter_features_list = [compute_features_and_labels(y,sr,region,label_type='laughter',source_file_id=source_file_id,file_index=index) for index, region in enumerate(laughter_regions)]
    speech_regions = [get_random_speech_region_from_file(t_file, get_length_from_regions_list(laughter_regions)) for i in xrange(1)]  #change 1 to get more speech than laughs
    speech_features_list = [compute_features_and_labels(y,sr,region,label_type='speech',source_file_id=source_file_id,file_index=index) for index, region in enumerate(speech_regions)]
    
    laughter_output_file = output_dir + "laughter_" + source_file_id + "_" + a_or_b + ".pkl"
    speech_output_file = output_dir + "speech_" + source_file_id + "_" + a_or_b + ".pkl"
    
    with open(laughter_output_file, "wb") as f:
        pickle.dump(laughter_features_list, f)

    with open(speech_output_file, "wb") as f:
        pickle.dump(speech_features_list, f)

def compute_all_features(transcription_file_list, output_dir, a_or_b):
    for index, t_file in enumerate(transcription_file_list):
        print "Processing %d out of %d transcription files." % (index+1, len(transcription_file_list))
        try:
            compute_and_store_features_and_labels(t_file, output_dir, a_or_b)
        except:
						print "File %d Failed" % (index + 1)

def parse_inputs():
	process = True

	try:
		t_root = sys.argv[1]
	except:
		print "Enter the switchboard transcriptions root dir as the first argument"
		process = False

	try:
		a_root = sys.argv[2]
	except:
		print "Enter the switchboard audio root directory as the second argument"
		process = False

	try:
		train_output_dir = sys.argv[3]
	except:
		print "Enter the training set output directory as the third argument"
		process = False
	
	try:
		validation_output_dir = sys.argv[4]
	except:
		print "Enter the validation set output directory as the fourth argument"
		process = False

	try:
		test_output_dir = sys.argv[5]
	except:
		print "Enter the test set output directory as the fourth argument"
		process = False

	if process:
		return (t_root, a_root, train_output_dir, validation_output_dir, test_output_dir)
	else:
		return False



# Usage: python compute_features.py <switchboard_transcriptions_dir> <switchboard_audio_dir> <train_output_dir> <val_output_dir> <test_output_dir>

if __name__ == '__main__':
	if parse_inputs():
		t_root, a_root, train_output_dir, validation_output_dir, test_output_dir = parse_inputs()
	
		# Get transcriptions root dir

		all_audio_files = get_all_audio_files(a_root)
		train_folders, val_folders, test_folders = get_train_val_test_folders(t_root)

		a_or_b = 'A'

		for a_or_b in ['A', 'B']:

			print "Laughter instances in training data: %d" % (count_laughter_instances_in_corpus(train_folders, a_or_b))
			print "Laughter instances in validation data: %d" % (count_laughter_instances_in_corpus(val_folders, a_or_b))
			print "Laughter instances in test data: %d" % ( count_laughter_instances_in_corpus(test_folders, a_or_b))
			print
			print "Files containing laughter in training data: %d" % (count_transcription_files_with_laughter_in_corpus(train_folders, a_or_b))
			print "Files containing laughter in validation data: %d" % (count_transcription_files_with_laughter_in_corpus(val_folders, a_or_b))
			print "Files containing laughter in test data: %d" % (count_transcription_files_with_laughter_in_corpus(test_folders, a_or_b))
			print

			train_audio_files = get_audio_files_from_transcription_files(get_all_transcriptions_files(train_folders, a_or_b))
			val_audio_files = get_audio_files_from_transcription_files(get_all_transcriptions_files(val_folders, a_or_b))
			test_audio_files = get_audio_files_from_transcription_files(get_all_transcriptions_files(test_folders, a_or_b))

			train_transcription_files = get_transcription_files_with_laughter_in_corpus(train_folders, a_or_b)
			val_transcription_files = get_transcription_files_with_laughter_in_corpus(val_folders, a_or_b)
			test_transcription_files = get_transcription_files_with_laughter_in_corpus(test_folders, a_or_b)

			train_audio_files = get_audio_files_from_transcription_files(train_transcription_files)
			val_audio_files = get_audio_files_from_transcription_files(val_transcription_files)
			test_audio_files = get_audio_files_from_transcription_files(test_transcription_files)
			print
			print "Training on %d dialogues" % len(train_audio_files)
			print "Validating on %d dialogues" % len(val_audio_files)
			print "Testing on %d dialogues" % len(test_audio_files)

		
			print "Computing Features for Training Data..."
			compute_all_features(train_transcription_files, train_output_dir, a_or_b)

			print "Computing Features for Validation Data..."
			compute_all_features(val_transcription_files, validation_output_dir, a_or_b)

			print "Computing Features for Test Data..."
			compute_all_features(test_transcription_files, test_output_dir, a_or_b)

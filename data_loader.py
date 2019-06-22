from tqdm import tqdm
from compute_features import *
import numpy as np, torch
from torch.utils import data
import librosa, time, numpy as np
from joblib import Parallel, delayed
import itertools
from sklearn.utils import shuffle


switchboard_audio_root = a_root = '/data/corpora/switchboard-1/97S62/'
switchboard_transcriptions_root = t_root = '/data/corpora/switchboard-1/swb_ms98_transcriptions/'
all_audio_files = get_all_audio_files(a_root)

train_folders, val_folders, test_folders = get_train_val_test_folders(t_root)

train_transcription_files_A, train_audio_files = get_audio_files_from_transcription_files(get_all_transcriptions_files(train_folders, 'A'), all_audio_files)
train_transcription_files_B, _ = get_audio_files_from_transcription_files(get_all_transcriptions_files(train_folders, 'B'), all_audio_files)

val_transcription_files_A, val_audio_files = get_audio_files_from_transcription_files(get_all_transcriptions_files(val_folders, 'A'), all_audio_files)
val_transcription_files_B, _ = get_audio_files_from_transcription_files(get_all_transcriptions_files(val_folders, 'B'), all_audio_files)

test_transcription_files_A, test_audio_files = get_audio_files_from_transcription_files(get_all_transcriptions_files(test_folders, 'A'), all_audio_files)
test_transcription_files_B, _ = get_audio_files_from_transcription_files(get_all_transcriptions_files(test_folders, 'B'), all_audio_files)


def parse_laughter_and_speech_files(transcription_files_a, transcription_files_b, all_audio_files):
	all_speech_regions = []
	all_laughter_regions = []
	for i in tqdm(range(len(transcription_files_a))):
		text_A = transcription_files_a[i]
		text_B = transcription_files_b[i]
		audio_file = get_audio_file_from_transcription_file(text_A, all_audio_files)
		laughter_regions_A = get_laughter_regions_from_file(text_A)
		laughter_regions_B = get_laughter_regions_from_file(text_B)
		laughter_regions = combine_overlapping_regions(laughter_regions_A, laughter_regions_B)
		speech_regions = []
		for laughter_region in laughter_regions:
			region_length = laughter_region[1] - laughter_region[0]
			audio_length = get_length_from_transcription_file(text_A)
			speech_regions.append(get_random_speech_region_from_files([text_A, text_B], audio_length, region_length, all_audio_files))
		all_laughter_regions.append(laughter_regions)
		all_speech_regions.append(speech_regions)
	return all_laughter_regions, all_speech_regions

train_laughter_regions, train_speech_regions = parse_laughter_and_speech_files(
	train_transcription_files_A, train_transcription_files_B, all_audio_files)

val_laughter_regions, val_speech_regions = parse_laughter_and_speech_files(
	val_transcription_files_A, val_transcription_files_B, all_audio_files)

test_laughter_regions, test_speech_regions = parse_laughter_and_speech_files(
	test_transcription_files_A, test_transcription_files_B, all_audio_files)

# librosa.load() but return only the signal, not (y, sr)
def librosa_load_without_sr(f, normalize=True, sr=None):
	y, sr = librosa.load(f, sr=sr)
	if normalize:
		m = np.max(np.abs(y)); if m > 0: y /= m
	return y

# Runs librosa.load() on a list of files in parallel, returns [y1, y2, ...]
def parallel_load_audio_batch(files, n_processes, sr=None):
	return Parallel(n_jobs=n_processes)(
		delayed(librosa_load_without_sr)(f,sr=sr) for f in tqdm(files))

def parallel_mel_spectrogram(signals, n_processes, sr=None, fps=100):
	return Parallel(n_jobs=n_processes)(
		delayed(librosa.feature.melspectrogram)(y, sr, 
			hop_length=int(sr/fps)) for y in tqdm(signals))

# Slice 1d signal into fixed sized segments
# input frame_length in seconds
# e.g. slice into 0.5 second segments
def slice_1d_signal(signal, seconds, hop_seconds, sr, randomize_phase=True):
	frame_length = int(seconds*sr)
	hop_length = int(hop_seconds*sr)
	if randomize_phase:
		signal = signal[np.random.randint(frame_length):]
	return librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length).T

# S is of shape (features, frames), librosa default
def slice_spectrogram(S, seconds, hop_seconds, sr, fps):
	hop_frames = int(fps * hop_seconds)
	start_frame = np.random.randint(hop_frames)
	max_slices = int(S.shape[1]/hop_frames)
	slices = librosa.util.index_to_slice(np.arange(start_frame, S.shape[1], hop_frames), idx_min=start_frame, idx_max=max_slices*hop_frames)
	sliced_specs = [S[:,s] for s in slices]
	sliced_specs = [librosa.util.pad_center(s,hop_frames,axis=1) if s.shape[1] < hop_frames else s for s in sliced_specs]
	arr = np.array(sliced_specs)
	assert(arr.shape[1] == 128)
	assert(arr.shape[2] == 10)
	return sliced_specs

class SwitchboardBatchDataset(torch.utils.data.Dataset):
	""" A class to load batches of audio files in Torch 
		and return the relevant laughter and speech clips
		Example Usage: 
		1. train_dataset = AudioBatchDataset(
			train_audio_files, train_labels, batch_size=32, n_processes=10)
		2. training_generator = data.DataLoader(train_dataset)
		3. for signals, labels in training_generator:
			... run training code ...
	"""
	def __init__(self, filepaths, all_laughter_regions, all_speech_regions, batch_size=32, n_processes=1, sr=8000):
		self.filepaths = filepaths
		self.batch_size = batch_size
		self.n_processes = n_processes
		self.all_laughter_regions = all_laughter_regions
		self.all_speech_regions = all_speech_regions
		self.sr = sr
	def __len__(self):
		# The length as the number of batches
		return int(np.ceil(float(len(self.filepaths)) / self.batch_size))
	def __getitem__(self, index):
		# Get a batch
		files = self.filepaths[index:index+self.batch_size]
		signals = parallel_load_audio_batch(files, self.n_processes)
		laughter_instances = self.all_laughter_regions[index:index+self.batch_size]
		speech_instances = self.all_speech_regions[index:index+self.batch_size]
		positive_examples = []
		negative_examples = []
		for i in range(len(files)):
			sig = signals[i]
			for l in laughter_instances[i]:
				positive_examples.append(clip_audio_region(signals[i], self.sr, l[0], l[1]))
			for s in speech_instances[i]:
				negative_examples.append(clip_audio_region(signals[i], self.sr, s[0], s[1]))
		return positive_examples, negative_examples
		#all_examples = positive_examples+negative_examples
		#labels = list(np.ones(len(positive_examples))) + list(np.zeros(len(negative_examples)))
		#return  all_examples, labels


train_dataset = SwitchboardBatchDataset(train_audio_files, train_laughter_regions, train_speech_regions, batch_size=128, n_processes=10)
train_generator = torch.utils.data.DataLoader(train_dataset)

val_dataset = SwitchboardBatchDataset(val_audio_files, val_laughter_regions, val_speech_regions, batch_size=128, n_processes=10)
val_generator = torch.utils.data.DataLoader(val_dataset)

test_dataset = SwitchboardBatchDataset(test_audio_files, test_laughter_regions, test_speech_regions, batch_size=128, n_processes=10)
test_generator = torch.utils.data.DataLoader(test_dataset)


X_pos, X_neg = iter(train_generator).next()

def get_spectrograms_for_batch(X):
	sigs = [x[0].numpy() for x in X]
	specs = parallel_mel_spectrogram(sigs, n_processes=10, sr=sr, fps=100)
	all_slices = [slice_spectrogram(s, 0.5, 0.1, 8000, 100) for s in tqdm(specs)]
	combined_specs = np.array(list(itertools.chain.from_iterable(all_slices)))
	return combined_specs

def get_model_inputs_and_labels_for_batch(X_pos, X_neg):
	positive_specs = get_spectrograms_for_batch(X_pos)
	positive_labels = np.ones(len(positive_specs))
	negative_specs = get_spectrograms_for_batch(X_neg)
	negative_labels = np.zeros(len(negative_specs))
	X = np.concatenate([positive_specs,negative_specs])
	y = np.concatenate([positive_labels,negative_labels])
	X, y = shuffle(X,y)
	return X, y

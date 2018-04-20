import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import numpy as np
import scipy.signal as signal
import os
import sys
import librosa
from python_speech_features import mfcc
from python_speech_features import delta
import keras
from keras.models import load_model

import compute_features

def frame_to_time(frame_index):
	return(frame/100.)

def seconds_to_frames(s):
	return(int(s*100))

def collapse_to_start_and_end_frame(instance_list):
    return (instance_list[0], instance_list[-1])

def frame_span_to_time_span(frame_span):
    return (frame_span[0] / 100., frame_span[1] / 100.)

def seconds_to_samples(s,sr):
    return s*sr

def format_features(mfcc_feat, delta_feat,index, window_size=37):
    return np.append(mfcc_feat[index-window_size:index+window_size],delta_feat[index-window_size:index+window_size])

def cut_laughter_segments(instance_list,y,sr):
	new_audio = []
	for start, end in instance_list:
		sample_start = int(seconds_to_samples(start,sr))
		sample_end = int(seconds_to_samples(end,sr))
		clip = y[sample_start:sample_end]
		new_audio = np.concatenate([new_audio,clip])
	return new_audio

def get_instances_from_rows(rows):
	return [(float(row.split(' ')[1]),float(row.split(' ')[2])) for row in rows]

def lowpass(sig, filter_order = 2, cutoff = 0.01):
	#Set up Butterworth filter
	filter_order  = 2
	B, A = signal.butter(filter_order, cutoff, output='ba')

	#Apply the filter
	return(signal.filtfilt(B,A, sig))

def get_laughter_instances(probs, threshold = 0.5, min_length = 0.2):
	instances = []
	current_list = []
	for i in xrange(len(probs)):
		if np.min(probs[i:i+1]) > threshold:
			current_list.append(i)
		else:
			if len(current_list) > 0:
				instances.append(current_list)
				current_list = []
	for i in instances:
		print i
		print
	instances = [frame_span_to_time_span(collapse_to_start_and_end_frame(i)) for i in instances if len(i) > min_length]
	return instances

def get_feature_list(y,sr,window_size=37):
	mfcc_feat = compute_features.compute_mfcc_features(y,sr)
	delta_feat = compute_features.compute_delta_features(mfcc_feat)
	zero_pad_mfcc = np.zeros((window_size,mfcc_feat.shape[1]))
	zero_pad_delta = np.zeros((window_size,delta_feat.shape[1]))
	padded_mfcc_feat = np.vstack([zero_pad_mfcc,mfcc_feat,zero_pad_mfcc])
	padded_delta_feat = np.vstack([zero_pad_delta,delta_feat,zero_pad_delta])
	feature_list = []
	for i in range(window_size, len(mfcc_feat) + window_size):
		feature_list.append(format_features(padded_mfcc_feat, padded_delta_feat, i, window_size))
	feature_list = np.array(feature_list)
	return feature_list

def format_outputs(instances, wav_paths):
	outs = []
	for i in xrange(len(instances)):
		outs.append({'filename': wav_paths[i], 'start': instances[i][0], 'end': instances[i][1]})
	return outs

def segment_laughs(input_path,model_path,output_path,threshold=0.5,min_length=0.2):
	print; print 'Loading audio file...'
	y,sr = librosa.load(input_path,sr=8000)
	full_res_y, full_res_sr = librosa.load(input_path,sr=44100)
	
	print; print 'Looking for laughter...'; print
	model = load_model(model_path)
	feature_list = get_feature_list(y,sr)

	probs = model.predict_proba(feature_list)
	probs = probs.reshape((len(probs),))#.reshape((len(mfcc_feat),))
	filtered = lowpass(probs)
	instances = get_laughter_instances(filtered, threshold=threshold, min_length=min_length)

	if len(instances) > 0:
		wav_paths = []
		maxv = np.iinfo(np.int16).max

		for index, instance in enumerate(instances):
			laughs = cut_laughter_segments([instance],full_res_y,full_res_sr)
			wav_path = output_path + "/laugh_" + str(index) + ".wav"
			librosa.output.write_wav(wav_path, (laughs * maxv).astype(np.int16), full_res_sr)
			wav_paths.append(wav_path)

		return(format_outputs(instances, wav_paths))

	else:
		return []
	

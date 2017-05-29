import numpy as np
import pickle
import os
import sys
from sklearn.utils import shuffle

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import keras.optimizers
from keras.models import load_model
import keras.regularizers
from keras.regularizers import l2, l1

# Methods for loading data from .pkl files created in computed_features.py

def load_hash(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)

def get_laughter_and_speech_clips(directory):
	laughter_files = [directory + f for f in os.listdir(directory) if 'laughter' in f]
	speech_files = [directory + f for f in os.listdir(directory) if not 'laughter' in f]

	laughter_data = [load_hash(f) for f in laughter_files]

	laughter_clips = []
	for f in laughter_data:
		for clip in f:
			laughter_clips.append(clip)
        
	speech_data = [load_hash(f) for f in speech_files]
	speech_clips = []
	for f in speech_data:
		for clip in f:
			speech_clips.append(clip)
    
	return (laughter_clips, speech_clips)



def format_laughter_inputs(clip):
	mfcc_feat = clip['mfcc']
	delta_feat = clip['delta']
	labels = clip['labels']
	laughter_frame_indices = np.nonzero(labels)[0]
	X = None
	for index in laughter_frame_indices:
		features = np.append(mfcc_feat[index-window_size:index+window_size],delta_feat[index-window_size:index+window_size])
		#features = np.append(mfcc_feat[index-window_size:index+window_size][:,1:13],delta_feat[index-window_size:index+window_size][:,1:13])
		if X is None:
			X = features
		else:
			X = np.vstack([X,features])
	return (X,np.ones(len(laughter_frame_indices)))

def format_speech_inputs(clip):
    mfcc_feat = clip['mfcc']
    delta_feat = clip['delta']
    labels = clip['labels']
    speech_frame_indices = np.array(list(xrange(len(labels))))[window_size:-window_size]
    X = []
    for index in speech_frame_indices:
        features = np.append(mfcc_feat[index-window_size:index+window_size],delta_feat[index-window_size:index+window_size])
        #features = np.append(mfcc_feat[index-window_size:index+window_size][:,1:13],delta_feat[index-window_size:index+window_size][:,1:13])
        X.append(features)
    return (np.array(X),np.zeros(len(speech_frame_indices)))

def format_laughter_clips(laughter_clips):
    formatted_laughter_clips = []
    for index, clip in enumerate(laughter_clips):
        if index % 500 == 0: print "formatting %d out of %d" % (index, len(laughter_clips))
        formatted_laughter_clips.append(format_laughter_inputs(clip))
    return formatted_laughter_clips
    
def format_speech_clips(speech_clips):
    formatted_speech_clips = []
    for index, clip in enumerate(speech_clips):
        if index % 500 == 0: print "formatting %d out of %d" % (index, len(speech_clips))
        formatted_speech_clips.append(format_speech_inputs(clip))
    return formatted_speech_clips

def get_data_and_labels_from_dir(directory):
    laughter_clips, speech_clips = get_laughter_and_speech_clips(directory)
    formatted_laughter_clips = format_laughter_clips(laughter_clips)
    formatted_speech_clips = format_speech_clips(speech_clips)
    train_data, train_labels = format_data_and_labels(formatted_laughter_clips, formatted_speech_clips)
    return (train_data, train_labels)

def format_data_and_labels(formatted_laughter_clips, formatted_speech_clips):
    train_data = []; train_labels = []
    for j in xrange(len(formatted_laughter_clips)):
        #print "Processing %d of %d" % (j,len(formatted_laughter_clips))
        clip, label = formatted_laughter_clips[j]
        if not clip is None and not label is None:
            for i in xrange(len(clip)):
                train_data.append(clip[i])
                train_labels.append(label[i])

    for j in xrange(len(formatted_speech_clips)):
        #print "Processing %d of %d" % (j,len(formatted_speech_clips))
        clip, label = formatted_speech_clips[j]
        if not clip is None and not label is None:
            for i in xrange(len(clip)):
                train_data.append(clip[i])
                train_labels.append(label[i])
                
    return (train_data, train_labels)

def divide_data_and_labels_into_parts(train_data,train_labels,part_size=5):
    train_data_parts = []
    train_label_parts = []
    i = 0
    while i < len(train_data) - part_size:
        train_data_parts.append(train_data[i:i+part_size])
        train_label_parts.append(train_labels[i:i+part_size])
        i += part_size
    return (train_data_parts, train_label_parts)

def get_data_subset(train_data_parts, train_label_parts, start, end):
    X = np.vstack(train_data_parts[start:end])
    y = np.hstack(train_label_parts[start:end])
    return X, y


# Methods for training and evaluating model

def initialize_model():
    model = Sequential()
    model.add(Dense(600, use_bias=True,input_dim=1924))#1924
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(100, use_bias=True,input_dim=1924))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return model    

def train_on_parts(train_data_parts, train_label_parts, name):
    train_data_parts, train_label_parts = shuffle(train_data_parts, train_label_parts, random_state=0)
    i = 0
    accs = []
    while i < len(train_data_parts):
        #print i
        X_subset, y_subset = get_data_subset(train_data_parts, train_label_parts, i, i+2000)
        model.fit(X_subset,y_subset,shuffle=True,batch_size = 500, epochs=1,verbose=False)
        acc = model.evaluate(X_subset, y_subset,verbose=False)[1]
        accs.append(acc)
        #print np.mean(accs)
        i += 2000
    print "%s accuracy %f" % (name, np.mean(accs))

def evaluate_on_parts(data_parts, label_parts, name):
    #train_data_parts, train_label_parts = shuffle(train_data_parts, train_label_parts, random_state=0)
    i = 0
    accs = []
    while i < len(data_parts):
        #if i % 10000 == 0: print i
        X_subset, y_subset = get_data_subset(data_parts, label_parts, i, i+100)
        #model.fit(X_subset,y_subset,shuffle=True,batch_size = 2000, epochs=1,verbose=False)
        acc = model.evaluate(X_subset, y_subset,verbose=False)[1]
        accs.append(acc)
        i += 100
    print "%s accuracy %f " % (name, np.mean(accs))
    return (np.mean(accs))

def parse_inputs():
	process = True

	try:
		train_dir = sys.argv[1]
	except:
		print "Enter the training set input directory as the first argument"
		process = False

	try:
		val_dir = sys.argv[2]
	except:
		print "Enter the validation set input directory as the second argument"
		process = False

	try:
		test_dir = sys.argv[3]
	except:
		print "Enter the test set input directory as the third argument"
		process = False

	try:
		stored_model_name = sys.argv[4]
	except:
		print "Enter the name for your saved model as the fourth argument"
		process = False
	
	if process:
		return (train_dir, val_dir, test_dir, stored_model_name)
	else:
		return False

# Usage: python train_model.py <training_input_dir> <validation_input_dir> <test_input_dir> <stored_model_name>

if __name__ == '__main__':
	if parse_inputs():
		train_dir, val_dir, test_dir, stored_model_name  = parse_inputs()
		window_size = 37 # window of 37 frames each to the left/right of the target frame
		

		print "Formatting Training Data..."
		print
		# format train set
		laughter_clips, speech_clips = get_laughter_and_speech_clips(train_dir)
		# Remove some clips that were failing - TODO fix this
		del laughter_clips[677]
		del laughter_clips[6079]
		del laughter_clips[7235]
		formatted_laughter_clips = format_laughter_clips(laughter_clips)
		formatted_speech_clips = format_speech_clips(speech_clips)
		train_data, train_labels = format_data_and_labels(formatted_laughter_clips, formatted_speech_clips)
		train_data_parts, train_label_parts = divide_data_and_labels_into_parts(train_data,train_labels,part_size=1)

		print "Formatting Validation Data..."
		print
		# format validation set
		val_laughter_clips, val_speech_clips = get_laughter_and_speech_clips(val_dir)
		val_formatted_laughter_clips = format_laughter_clips(val_laughter_clips)
		val_formatted_speech_clips = format_speech_clips(val_speech_clips)
		val_data, val_labels = format_data_and_labels(val_formatted_laughter_clips, val_formatted_speech_clips)
		val_data_parts, val_label_parts = divide_data_and_labels_into_parts(val_data,val_labels,part_size=1)
		
		print "Formatting Test Data..."
		print
		# format test set
		test_laughter_clips, test_speech_clips = get_laughter_and_speech_clips(test_dir)
		test_formatted_laughter_clips = format_laughter_clips(test_laughter_clips)
		test_formatted_speech_clips = format_speech_clips(test_speech_clips)
		test_data, test_labels = format_data_and_labels(test_formatted_laughter_clips, test_formatted_speech_clips)
		test_data_parts, test_label_parts = divide_data_and_labels_into_parts(test_data,test_labels,part_size=1)

		model = initialize_model()
		best_val_acc = 0

		for epoch in xrange(20):
			print "Epoch %d" % (epoch)
			train_on_parts(train_data_parts, train_label_parts, "Training")
			val_acc = evaluate_on_parts(val_data_parts, val_label_parts, "Validation")
			test_acc = evaluate_on_parts(test_data_parts, test_label_parts, "Test")
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				model.save(stored_model_name)
		


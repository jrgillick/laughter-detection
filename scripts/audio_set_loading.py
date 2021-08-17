##################### Audioset (Laughter Detection)  #############################
import sys, librosa
sys.path.append('../utils/')
import audio_utils, text_utils
from download_audio_set_mp3s import *
from sklearn.utils import shuffle

audioset_train_path='../data/audioset/unbalanced_train_laughter_audio'
audioset_test_path='../data/audioset/eval_laughter_audio'
audioset_train_labels_path='../data/audioset/unbalanced_train_segments.csv'
audioset_test_labels_path='../data/audioset/eval_segments.csv'

def get_audioset_laughter_train_val_test_files(
    audioset_train_path=audioset_train_path,
    audioset_test_path=audioset_test_path,
    audioset_train_labels_path=audioset_train_labels_path,
    audioset_test_labels_path=audioset_test_labels_path):
    audioset_train_files = librosa.util.find_files(audioset_train_path, ext=['mp3'])
    cutoff = int(0.8*len(audioset_train_files))
    audioset_val_files = audioset_train_files[cutoff:]
    audioset_train_files = audioset_train_files[0:cutoff]
    audioset_test_files = librosa.util.find_files(audioset_test_path, ext=['mp3'])
    return audioset_train_files, audioset_val_files, audioset_test_files

def get_audioset_ids(csv_file, mode):
    infolist = get_laughter_infolist(csv_file, mode=mode)
    return [l['yt_id'] for i, l in enumerate(infolist)]

# Get a dictionary that maps from an audioset file ID to a list of
# laughter class [<belly laugh, giggle, etc.]
def get_audioset_laughter_classes_dict(csv_files, return_type='vector'):
    d = {}
    if type(csv_files) != type([]): csv_files = [csv_files]
    for csv_file in csv_files:
        infolist = get_laughter_infolist(csv_file, mode='positive')
        ids = [l['yt_id'] for i, l in enumerate(infolist)]
        tag_strings = [l['tag_strings'] for l in infolist]
        assert(len(ids) == len(tag_strings))

        if return_type == 'vector':
            for i in range(len(ids)):
                d[ids[i]] = laugh_id_multihot(tag_strings[i])
        elif return_type == 'string':
            for i in range(len(ids)):
                d[ids[i]] = laugh_id_dict[tag_strings[i]]
        else:
            raise Exception("Invalid return_type")
    return d

def get_ytid_from_filepath(f):
    return os.path.splitext(os.path.basename(f))[0].split('yt_')[1]
    

# For binary laughter detection
def get_audioset_binary_labels(files, positive_ids,negative_ids):
    labels = []
    for f in files:
        fid = get_ytid_from_filepath(f)
        if fid in positive_ids:
            labels.append(1)
        elif fid in negative_ids:
            labels.append(0)
        else:
            raise Exception("Unfound Youtube ID")
    return labels

# for laughter type classification - e.g. giggle, belly laugh, etc.
def get_audioset_multiclass_labels(files):
    labels = []
    positive_ids = list(audioset_laughter_classes_dict.keys())
    for f in files:
        fid = get_ytid_from_filepath(f)
        if fid in positive_ids:
            labels.append(audioset_laughter_classes_dict[fid])
        else:
            labels.append(np.zeros(len(laugh_keys)))
    return labels

audioset_positive_laughter_ids = get_audioset_ids(
    audioset_train_labels_path, 'positive') + get_audioset_ids(
    audioset_test_labels_path, 'positive')

audioset_negative_laughter_ids = get_audioset_ids(
    audioset_train_labels_path, 'negative') + get_audioset_ids(
    audioset_test_labels_path, 'negative')

def get_random_1_second_snippets(audio_signals, samples_per_file=1, sr=8000):
    audios = []
    for j in range(samples_per_file):
        audio_times = [audio_utils.subsample_time(0, int(len(a)/sr), int(len(a)/sr),
            subsample_length=1., padding_length=0.) for a in audio_signals]    
        for i in range(len(audio_signals)):
            start_time = librosa.core.time_to_samples(audio_times[i][0], sr=sr)
            end_time = librosa.core.time_to_samples(audio_times[i][0] + audio_times[i][1], sr=sr)
            aud = audio_signals[i][start_time:end_time]
            audios.append(aud)
    return audios





########## For evaluation, let's redo the train/test split sizes and save results to a file to make it permanent ####

# audioset_positive_laughter_ids has all the laughter files in audioset.
# we don't need to use audioset's official train/dev split.
# So let's just combine all the files, then split.
# Reserve 1500 for test, 500 for dev, and make the rest training



# 1. Find all audio files
all_audioset_files = librosa.util.find_files(audioset_train_path) + librosa.util.find_files(audioset_test_path)

# 2. Find all the positive and negative files that were successfully downloaded
positive_audioset_files = []
negative_audioset_files = []

filepath_to_ytid = {}
for f in all_audioset_files:
    ytid = get_ytid_from_filepath(f)
    filepath_to_ytid[f] = ytid
    if ytid in audioset_positive_laughter_ids:
        positive_audioset_files.append(f)
    else:
        negative_audioset_files.append(f)
        
ytid_to_filepath = text_utils.make_reverse_vocab(filepath_to_ytid)
        
# 3. Trim the negative examples list to be the same size as the positives
negative_audioset_files = negative_audioset_files[0:len(positive_audioset_files)]

# 4. Now Shuffle all files with random seed
positive_audioset_files = sorted(positive_audioset_files)
np.random.seed(0)
positive_audioset_files = shuffle(positive_audioset_files)

negative_audioset_files = sorted(negative_audioset_files)
np.random.seed(0)
negative_audioset_files = shuffle(negative_audioset_files)

# 5. Filter our list of ID's to match the list of files that were successfully downloaded
audioset_positive_laughter_ids = [get_ytid_from_filepath(f) for f in positive_audioset_files]        
audioset_negative_laughter_ids = [get_ytid_from_filepath(f) for f in negative_audioset_files]

# 6. Make the splits on both files and ID's, now that all files and ID's are matching and shuffled in the same order
# Laughter files and ID's for test, dev, train
test_positive_laughter_files = positive_audioset_files[0:1500]
test_positive_laughter_ids = audioset_positive_laughter_ids[0:1500]

dev_positive_laughter_files = positive_audioset_files[1500:2000]
dev_positive_laughter_ids = audioset_positive_laughter_ids[1500:2000]

train_positive_laughter_files = positive_audioset_files[2000:]
train_positive_laughter_ids = audioset_positive_laughter_ids[2000:]

# Distractor files and ID's for test, dev, train
test_negative_laughter_files = negative_audioset_files[0:1500]
test_negative_laughter_ids = audioset_negative_laughter_ids[0:1500]

dev_negative_laughter_files = negative_audioset_files[1500:2000]
dev_negative_laughter_ids = audioset_negative_laughter_ids[1500:2000]

train_negative_laughter_files = negative_audioset_files[2000:]
train_negative_laughter_ids = audioset_negative_laughter_ids[2000:]
        
# 7. save txt files with the splits - only need to do once
"""
#Save IDS
with open('../data/audioset/splits/test_laughter_ids.txt', 'w') as f:
    f.write("\n".join(test_positive_laughter_ids))
    
with open('../data/audioset/splits/dev_laughter_ids.txt', 'w') as f:
    f.write("\n".join(dev_positive_laughter_ids))
    
with open('../data/audioset/splits/train_laughter_ids.txt', 'w') as f:
    f.write("\n".join(train_positive_laughter_ids))
    
with open('../data/audioset/splits/test_negative_ids.txt', 'w') as f:
    f.write("\n".join(test_negative_laughter_ids))
    
with open('../data/audioset/splits/dev_negative_ids.txt', 'w') as f:
    f.write("\n".join(dev_negative_laughter_ids))
    
with open('../data/audioset/splits/train_negative_ids.txt', 'w') as f:
    f.write("\n".join(train_negative_laughter_ids))
    
# Save Filepaths
with open('../data/audioset/splits/test_laughter_files.txt', 'w') as f:
    f.write("\n".join(test_positive_laughter_files))
    
with open('../data/audioset/splits/dev_laughter_files.txt', 'w') as f:
    f.write("\n".join(dev_positive_laughter_files))
    
with open('../data/audioset/splits/train_laughter_files.txt', 'w') as f:
    f.write("\n".join(train_positive_laughter_files))
    
with open('../data/audioset/splits/test_negative_files.txt', 'w') as f:
    f.write("\n".join(test_negative_laughter_files))
    
with open('../data/audioset/splits/dev_negative_files.txt', 'w') as f:
    f.write("\n".join(dev_negative_laughter_files))
    
with open('../data/audioset/splits/train_negative_files.txt', 'w') as f:
    f.write("\n".join(train_negative_laughter_files))
"""



# 8. Update the labels so they match the splits

audioset_test_files = test_positive_laughter_files + test_negative_laughter_files
audioset_dev_files = dev_positive_laughter_files + dev_negative_laughter_files
audioset_train_files = train_positive_laughter_files + train_negative_laughter_files

audioset_test_labels = get_audioset_binary_labels(
    audioset_test_files, positive_ids=audioset_positive_laughter_ids, negative_ids=audioset_negative_laughter_ids)
audioset_val_labels = get_audioset_binary_labels(
    audioset_dev_files, positive_ids=audioset_positive_laughter_ids, negative_ids=audioset_negative_laughter_ids)
audioset_dev_labels = audioset_val_labels # Just in case used somewhere :(
audioset_train_labels = get_audioset_binary_labels(
    audioset_train_files, positive_ids=audioset_positive_laughter_ids, negative_ids=audioset_negative_laughter_ids)

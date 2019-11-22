##################### Audioset (Laughter Detection)  #############################
import sys, librosa
sys.path.insert(0, '../../audio_set/')
from download_audio_set_mp3s import *

audioset_train_path='/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/unbalanced_train_laughter_audio'
audioset_test_path='/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/eval_laughter_audio'
audioset_train_labels_path='/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/unbalanced_train_segments.csv'
audioset_test_labels_path='/mnt/data0/jrgillick/projects/laughter-detection/data/audioset/eval_segments.csv'

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
def get_audioset_laughter_classes_dict(csv_file, return_type='vector'):
    infolist = get_laughter_infolist(csv_file, mode='positive')
    ids = [l['yt_id'] for i, l in enumerate(infolist)]
    tag_strings = [l['tag_strings'] for l in infolist]
    assert(len(ids) == len(tag_strings))
    d = {}
    if return_type == 'vector':
        for i in range(len(ids)):
            d[ids[i]] = laugh_id_multihot(tag_strings[i])
    elif return_type == 'string':
        for i in range(len(ids)):
            d[ids[i]] = laugh_id_dict[tag_strings[i]]
    else:
        raise Exception("Invalid return_type")
    return d



# For binary laughter detection
def get_audioset_binary_labels(files, positive_ids,negative_ids):
    def _get_ytid_from_filepath(f):
        return os.path.splitext(os.path.basename(f))[0].split('yt_')[1]
    labels = []
    for f in files:
        fid = _get_ytid_from_filepath(f)
        if fid in positive_ids:
            labels.append(1)
        elif fid in negative_ids:
            labels.append(0)
        else:
            raise Exception("Unfound Youtube ID")
    return labels

# for laughter type classification - e.g. giggle, belly laugh, etc.
def get_audioset_multiclass_labels(files):
    def _get_ytid_from_filepath(f):
        return os.path.splitext(os.path.basename(f))[0].split('yt_')[1]
    labels = []
    positive_ids = list(audioset_laughter_classes_dict.keys())
    for f in files:
        fid = _get_ytid_from_filepath(f)
        if fid in positive_ids:
            labels.append(audioset_laughter_classes_dict[fid])
        else:
            labels.append(np.zeros(len(laugh_keys)))
    return labels

audioset_train_files, audioset_val_files, audioset_test_files = get_audioset_laughter_train_val_test_files()

audioset_laughter_classes_dict = get_audioset_laughter_classes_dict(audioset_test_labels_path)

audioset_positive_laughter_ids = get_audioset_ids(
    audioset_train_labels_path, 'positive') + get_audioset_ids(
    audioset_test_labels_path, 'positive')

audioset_negative_laughter_ids = get_audioset_ids(
    audioset_train_labels_path, 'negative') + get_audioset_ids(
    audioset_test_labels_path, 'negative')

audioset_test_labels = get_audioset_binary_labels(
    audioset_test_files, positive_ids=audioset_positive_laughter_ids, negative_ids=audioset_negative_laughter_ids)
audioset_val_labels = get_audioset_binary_labels(
    audioset_val_files, positive_ids=audioset_positive_laughter_ids, negative_ids=audioset_negative_laughter_ids)
audioset_train_labels = get_audioset_binary_labels(
    audioset_train_files, positive_ids=audioset_positive_laughter_ids, negative_ids=audioset_negative_laughter_ids)

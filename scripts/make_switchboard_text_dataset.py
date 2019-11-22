#python scripts/make_switchboard_text_dataset.py --output_txt_file=/data/jrgillick/projects/laughter-detection/data/switchboard/train/train_1.txt --switchboard_audio_path=/data/corpora/switchboard-1/97S62/ --switchboard_transcriptions_path=/data/corpora/switchboard-1/swb_ms98_transcriptions/ --data_partition=train

#a_root = '/data/corpora/switchboard-1/97S62/'
#t_root = '/data/corpora/switchboard-1/swb_ms98_transcriptions/'

import sys, time, librosa, os, argparse, pickle
from tqdm import tqdm
sys.path.append('/mnt/data0/jrgillick/projects/audio-feature-learning/')
import dataset_utils, audio_utils, data_loaders
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()

##################################################################
######################  Get input arguments ######################
##################################################################

# Path to store the parsed label times and inputs for Switchboard
parser.add_argument('--output_txt_file', type=str, required=True)
# Path to the root folder containing the switchboard audio
parser.add_argument('--switchboard_audio_path', type=str, required=True)
# Path to the root folder containing the switchboard transcriptions
parser.add_argument('--switchboard_transcriptions_path', type=str, required=True)
# Choose train/dev/test data
parser.add_argument('--data_partition', type=str, required=True)
# Number of passes through the data using negative sampling
parser.add_argument('--num_passes', type=str)
parser.add_argument('--load_audio_path', type=str)

args = parser.parse_args()

output_txt_file = args.output_txt_file
a_root = args.switchboard_audio_path
t_root = args.switchboard_transcriptions_path
data_partition = args.data_partition
num_passes = args.num_passes if args.num_passes is not None else 1
load_audio = args.load_audio_path is not None
load_audio_path = args.load_audio_path
"""
def make_text_dataset(t_files_a, t_files_b, audio_files,num_passes=1):
    big_list = []
    assert(len(t_files_a)==len(t_files_b) and len(t_files_a)==len(audio_files))
    for p in range(num_passes):
        for i in tqdm(range(len(t_files_a))):
            laugh_regions, speech_regions = dataset_utils.get_laughter_regions_and_speech_regions(
                t_files_a[i],t_files_b[i],audio_files[i])
            audio_file = audio_files[i]
            for r in laugh_regions:
                line = list(r) + [audio_files[i]] + [1]
                big_list.append('\t'.join([str(l) for l in line]))
            for r in speech_regions:
                line = list(r) + [audio_files[i]] + [0]
                big_list.append('\t'.join([str(l) for l in line]))
    return big_list
"""

def make_text_dataset(t_files_a, t_files_b, audio_files,num_passes=1,n_processes=8):
    big_list = []
    assert(len(t_files_a)==len(t_files_b) and len(t_files_a)==len(audio_files))
    for p in range(num_passes):
        lines_per_file = Parallel(n_jobs=n_processes)(
            delayed(dataset_utils.get_laughter_speech_text_lines)(t_files_a[i],
                    t_files_b[i], audio_files[i]) for i in tqdm(range(len(t_files_a))))
        big_list += audio_utils.combine_list_of_lists(lines_per_file)
    return big_list

def librosa_load_without_sr(f, sr=None,offset=None,duration=None):
    return librosa.load(f, sr=sr,offset=offset,duration=duration)[0]
# Runs librosa.load() on a list of files in parallel, returns [y1, y2, ...]
def parallel_load_audio_batch(files,n_processes,sr=None,offsets=None,
    durations=None):
    if offsets is not None and durations is not None:
        return Parallel(n_jobs=n_processes)(
            delayed(librosa_load_without_sr)(files[i],sr=sr,offset=offsets[i],
                duration=durations[i]) for i in tqdm(range(len(files))))
    else:
        return Parallel(n_jobs=n_processes)(
            delayed(librosa_load_without_sr)(f,sr=sr) for f in tqdm(files))
    
def get_audios_from_text_data(data_file, h, sr=8000):
    audios = []
    df = pd.read_csv(data_file,sep='\t',header=None,
        names=['offset','duration','audio_path','label'])
    audio_paths = list(df.audio_path)
    offsets = list(df.offset)
    durations = list(df.duration)
    for i in tqdm(range(len(audio_paths))):
        aud = h[audio_paths[i]][int(offsets[i]*sr):int((offsets[i]+durations[i])*sr)]
        audios.append(aud)
    return audios

##################################################################
#######################  Load Switchboard Data  ##################
##################################################################
                                
all_audio_files = librosa.util.find_files(a_root,ext='sph')

train_folders, val_folders, test_folders = dataset_utils.get_train_val_test_folders(t_root)

train_transcription_files_A, train_audio_files = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(train_folders, 'A'), all_audio_files)
train_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(train_folders, 'B'), all_audio_files)

val_transcription_files_A, val_audio_files = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(val_folders, 'A'), all_audio_files)
val_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(val_folders, 'B'), all_audio_files)

test_transcription_files_A, test_audio_files = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(test_folders, 'A'), all_audio_files)
test_transcription_files_B, _ = dataset_utils.get_audio_files_from_transcription_files(dataset_utils.get_all_transcriptions_files(test_folders, 'B'), all_audio_files)

##################################################################
##############################  Run  #############################
##################################################################

if data_partition == 'train':
    print('train')
    t_files_a = train_transcription_files_A
    t_files_b = train_transcription_files_B
    a_files = train_audio_files
elif data_partition == 'val':
    print('val')
    t_files_a = val_transcription_files_A
    t_files_b = val_transcription_files_B
    a_files = val_audio_files
elif data_partition == 'test':
    print('test')
    t_files_a = test_transcription_files_A
    t_files_b = test_transcription_files_B
    a_files = test_audio_files
else:
    raise Exception("data_partition must be one of `train`, `val`, or `test`")    
    
lines = make_text_dataset(t_files_a, t_files_b, a_files,num_passes=num_passes)
                                
with open(output_txt_file, 'w')  as f:
    f.write('\n'.join(lines))

"""
if load_audio:
    #df = pd.read_csv(output_txt_file,sep='\t',header=None,
    #        names=['offset','duration','audio_path','label'])
    #audios = parallel_load_audio_batch(df.audio_path,n_processes=16,
    #                                   offsets=df.offset,durations=df.duration)
    print("Loading switchboard audio files...")
    t0 = time.time()
    with open(load_audio_path, "rb") as f: # Loads all switchboard audio files
        h = pickle.load(f)
    print("Loaded in ", time.time()-t0, " seconds.") 
    audios = get_audios_from_text_data(output_txt_file, h)
    output_audio_file = output_txt_file.replace('.txt','.pkl')
    with open(output_audio_file, 'wb') as f:
        pickle.dump(audios, f)
"""


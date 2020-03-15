import numpy as np, os, json, pandas as pd, audioread
from tqdm import tqdm

def get_audio_file_length(path):
    f = audioread.audio_open(path)
    l = f.duration
    f.close()
    return l

annotations_folder = '../data/audioset/annotations/'
main_annotations_file = annotations_folder + '/raw_annotations/laughter_annotations.csv'
additional_annotations_files = [
    annotations_folder + 'raw_annotations/annotations_3.txt',
    annotations_folder + 'raw_annotations/annotations_4.txt',
    annotations_folder+ 'raw_annotations/annotations_5.txt',
    annotations_folder + 'raw_annotations/annotations_6.txt']

# Parsing main annotations file into a DataFrame
df = pd.read_csv(main_annotations_file, sep=',')
headers = list(df.keys())
df = df[df.Start.notnull()]
df.reset_index(inplace=True)

# Find rows with no laughter - these are marked 'NO Laughter' in the 'Start' column
rowz = []
for i in range(len(df.Start)):
    try:
        float(df.Start[i])
    except:
        rowz.append(i)
        df.at[i, 'Start'] = 0.0
        df.at[i, 'End'] = 0.0 # Not actually 0. Need to set this to length of audio file later.
print(f"Found {len(rowz)} of {len(df)} rows with no laughter")

df.reset_index(inplace=True,drop=True)

# Load additional files
def parse_audio_annotator_file(f):
    lines = open(f).read().split('\n')[0:-1]
    output_rows = []
    for i, line in enumerate(lines):
        h = {}
        try:
            row = json.loads(line)
        except:
            h['FileID'] = line.split('“')[1].split(".json")[0]
            h['Start'] = '0.000'; h['End'] = '0.000'
            output_rows.append(h)
            continue # If there were no annotations, this should be all non-laughter from 0 to 10 secs
        h['FileID'] = row[0]['file_name'].split('json/')[1].split('.json')[0]
        for annotation_number, annotation in enumerate(row):
            start_str = str(annotation['start'])[0:5]
            end_str = str(annotation['end'])[0:5]
            if annotation_number == 0:
                h['Start'] = start_str; h['End'] = end_str
            else:
                h[f'Start.{annotation_number}'] = start_str; h[f'End.{annotation_number}'] = end_str
            #row = ','.join([FileID, start_str, end_str])
        output_rows.append(h)
    return pd.DataFrame(output_rows)

for f in additional_annotations_files:
    new_df = parse_audio_annotator_file(f)
    df = pd.concat([df, new_df])
    df.reset_index(inplace=True,drop=True)

# Put columns back in the origin order
df = df[['FileID','Start','End','Start.1','End.1','Start.2','End.2','Start.3','End.3','Start.4','End.4']]

test_laughter_files = open('../data/audioset/splits/test_laughter_files.txt').read().split('\n')
test_laughter_ids = open('../data/audioset/splits/test_laughter_ids.txt').read().split('\n')

for i, FileID in tqdm(enumerate(list(df.FileID))):
    for f in test_laughter_files:
        if FileID in f:
            df.at[i, 'audio_path'] = f
            df.at[i, 'audio_length'] = str(get_audio_file_length(f))
            continue

# Find and then fix a couple of formatting anomalies
keys = list(df.columns)
for i in range(len(df)):
    for key in keys:
        if type(df.at[i,key]) is str and df.at[i,key].strip()=='':
            print(i, key)
            
df.at[287, 'End.1'] = df.at[287, 'Start.1']
df.at[302, 'End.2'] = df.at[302, 'Start.2']


# Find and then fix boundary errors where labeled regions extend slightly past the end of the audio clip
keys = [k for k in list(df.columns) if k not in ['FileID','audio_path', 'audio_length']]
boundary_errors = []
for i in range(len(df)):
    for key in keys:
        if float(df.at[i,key]) > float(df.at[i,'audio_length']):
            boundary_errors.append(float(df.at[i,key]) - float(df.at[i,'audio_length']))
            df.at[i,key] = df.at[i,'audio_length']

print(f"Boundary Errors: {len(boundary_errors)} | Mean Error: {np.mean(boundary_errors)} | Max Error: {np.max(boundary_errors)}")


df.to_csv('../data/audioset/annotations/clean_laughter_annotations.csv', index=None)
import numpy as np, os, json
import pandas as pd

main_annotations_file = 'raw_annotations/laughter_annotations.csv'
additional_annotations_files = ['raw_annotations/annotations_3.txt', 'raw_annotations/annotations_4.txt', 'raw_annotations/annotations_5.txt', 'raw_annotations/annotations_6.txt']

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
        df.at[i, 'End'] = 10.0
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

df.to_csv('clean_laughter_annotations.csv', index=None)
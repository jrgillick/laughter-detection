import pandas as pd
import numpy as np, os, sys
import warnings
warnings.simplefilter("ignore")
from eval_utils import *
from tqdm import tqdm

MIN_GAP = 0

path_to_baseline_code = '/mnt/data0/jrgillick/projects/laughter-detection-2018/laughter-detection/'
baseline_model_path = path_to_baseline_code + '/models/model.h5'
sys.path.append(path_to_baseline_code)
import laugh_segmenter as baseline_laugh_segmenter

annotations_df = pd.read_csv('../../data/audioset/annotations/clean_laughter_annotations.csv')
baseline_model = baseline_laugh_segmenter.load_model(baseline_model_path)

all_results = []
for i in tqdm(range(len(annotations_df))):
    h = get_baseline_results_per_annotation_index(
        baseline_model, annotations_df, baseline_laugh_segmenter, i, min_gap=MIN_GAP,
        use_filter=True,min_length=0.1)
    all_results.append(h)
    
results_df = pd.DataFrame(all_results)
results_df.to_csv("baseline_audioset_results.csv",index=None)
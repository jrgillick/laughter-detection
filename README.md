# laughter-detection

### UPDATE(August 2021):
* The library has been updated to use a newer detection model that is more accurate and more robust to background noise. It also now uses Pytorch instead of Tensorflow. An older version of the software from the 2018 paper is still available [here](https://github.com/jrgillick/laughter-detection/tree/v1.0).

## Overview

This library contains code and models to automatically detect and segment regions containing human laughter from an audio file. The [checkpoints](checkpoints/) folder contains models trained on the [Switchboard](https://catalog.ldc.upenn.edu/ldc97s62) data set.

This library also includes [annotations](data/audioset/annotations/clean_laughter_annotations.csv) for evaluating laughter detection in real-world environments using the [AudioSet](https://research.google.com/audioset/) dataset.

Code, Annotations and Models are described in the following papers:

- Jon Gillick, Wesley Deng, Kimiko Ryokai, and David Bamman, "Robust Laughter Detection in Noisy Environments" (2021), Interspeech 2021.
- Kimiko Ryokai, Elena Durán López, Noura Howell, Jon Gillick, and David Bamman (2018), "Capturing, Representing, and Interacting with Laughter," CHI 2018


## Installation

```sh
git clone https://github.com/jrgillick/laughter-detection.git
cd laughter-detection
pip install -r requirements.txt
```

# Dependencies
- Python - tested with version 3.6.1.
- [Librosa](http://librosa.github.io/librosa/) - tested with version 0.8.1.
- [Pytorch](https://pytorch.org/) - tested with version 1.3.1.
- A somewhat more comprehensive list can be found in [requirements.txt](requirements.txt).

# Usage
- To run interactively on Google Colab, use [this link](https://colab.research.google.com/github/jrgillick/laughter-detection/blob/master/laughter-detection-interactive.ipynb).
- To run laughter detection from the command line, use the [segment_laughter.py](segment_laughter.py) script.  
- For example: `python segment_laughter.py --input_audio_file=tst_wave.wav --output_dir=./tst_wave --save_to_textgrid=False --save_to_audio_files=True --min_length=0.2 --threshold=0.5`

  #### Parameters
  - The threshold parameter adjusts the minimum probability threshold for classifying a frame as laughter. The default is 0.5, but you can  experiment with settings between 0 and 1 to see what works best for your audio. Lower threshold values may give more false positives but may also recover a higher percentage of laughter from your file.

  - The min_length parameter sets the minimum length in seconds that a laugh needs to be in order to be identified. The default value is 0.2.
  - The full list of parameters can be found in [segment_laughter.py](segment_laughter.py).

  #### Output
  - The script prints out a list of time segments in seconds of the form (start, end) and saves them as a list of wav files into a folder at `<output_dir>`
  - Output can also be saved as a Textgrid file.

# Training
- Training is implemented in [train.py](train.py). This requires downloading data from either Switchboard or AudioSet, and running some of the pre-processing [scripts](scripts).

# Evaluation
- Code and results for the experiments in the Interspeech 2021 paper are included [here](scripts/Evaluation).


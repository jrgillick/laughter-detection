import torch, numpy as np, librosa
from torch.utils import data
import dataset_utils, audio_utils
from joblib import Parallel, delayed
import pandas as pd

class AudioDataset(torch.utils.data.Dataset):
    """ A class to load audio files and, optionally, labels.

        Two kinds of usage: "Basic", and "Subsampling".

        In "Basic" usage, a file corresponds to one "Example", so we assume
        one feature-set (and label-set) per file. This works for any dataset
        where a single file corresponds to a single example.  In the "Basic"
        setup, you can just use `feature_fn` and `label_fn` to get features
        and labels.  You could also use `feature_and_label_fn` instead if that
        is easier for you.

        Basic Usage:
        1. train_dataset = AudioDataset(train_audio_files, feature_fn=featurize_mfcc,
            label_paths=train_label_files, label_fn = get_labels)  OR
           train_dataset = AudioDataset(train_audio_files, feature_fn=featurize_mfcc,
            label_paths=train_label_files, labels = my_precomputed_labels)
        2. training_generator = torch.utils.data.DataLoader(train_dataset,
            num_workers=8, batch_size=8, shuffle=True, collate_fn=pad_sequences)
        3. for batch_index, batch_feats in enumerate(training_generator):
            if i_batch == 50:
              log or do something...
            feats, labels = batch_feats...

        For "Subsampling", we deal with the case when we want to extract
        multiple (and labels if desired) from a single file. Here, we must use
        `feature_and_label_fn` and `feature_fn`, as opposed to `feature_fn`
        and `label_fn`,  so that subsampling logic can be shared.
        Now, we expect features and labels returned by __get_item__ to be a
        list.  The corresponding  `collate_fn` you pass to the DataLoader
        needs to be able to handle this. We can do one type check to make
        sure __get_item__ at least returns a list, but you still need to
        remember to handle it correctly in your `collate_fn`.

        Subsampling Usage:
        1. train_dataset = AudioDataset(train_audio_files,
            feature_and_label_fn=sample_timit_features_and_labels,
            feature_fn = audio_utils.featurize_mfcc,
            label_paths=train_label_files,
            does_subsample=True)
        2. training_generator = torch.utils.data.DataLoader(train_dataset,
            num_workers=8, batch_size=8, shuffle=True, collate_fn=pad_sequences)
        3. for batch_index, batch_feats in enumerate(training_generator):
            if i_batch == 50:
              log or do something...
            feats, labels = batch_feats...


        Args:
        filepaths: A list of paths to audio files.
        feature_and_label_fn: needs to accept args filepath, label_path, feature_fn, **kwargs
        feature_fn: A feature function to apply to each file.
        label_paths: A list of paths to label files corresponding to the files in filepaths.
        label_fn: A function to get labels from files in label_paths.
        labels: Instead of label_paths, labels can be passed in directly.
    """
    def __init__(self, filepaths, feature_fn=None, label_paths=None,
        label_fn=None, labels=None, feature_and_label_fn=None,
        does_subsample=None, **kwargs):

        self.filepaths = filepaths
        self.feature_and_label_fn = feature_and_label_fn
        self.feature_fn = feature_fn
        self.label_paths = label_paths
        self.label_fn = label_fn
        self.labels = labels
        self.does_subsample = does_subsample


        # Validate input args
        if label_paths is not None and labels is not None:
            raise Exception("Don't pass both `label_paths` and `labels`")

        if not feature_and_label_fn and (bool(label_paths) ^ bool(label_fn)):
            raise Exception("Can't use only one of `label_paths` and `label_fn`")

        self.include_labels = bool(label_paths) or bool(labels) or self.does_subsample
        #print("Includes Labels: %s" % str(self.include_labels))
        #print("Uses Subsampling: %s" % str(does_subsample))

        # **kwargs can include params to pass along to feature_fn and label_fn
        # These should be passed in as 'feature_fn_args' and 'label_fn_args'
        # For example, feature_fn may accept an argument `window_size`, and
        # label_fn may accept an argument `vocab`.
        self.kwargs = kwargs or {}
        self.feature_fn_args = kwargs['feature_fn_args'] if  'feature_fn_args' \
            in kwargs else {}
        self.label_fn_args = kwargs['label_fn_args'] if  'label_fn_args' \
            in kwargs else {}

    def __len__(self):
        return len(self.filepaths)

    def get_label(self, index):
        if self.labels is not None:
            return self.labels[index]
        elif self.label_paths is not None:
            return self.label_fn(self.label_paths[index], **self.label_fn_args)
        else:
            raise Exception("No labels provided.")

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        if self.does_subsample:
            features_and_labels = self.feature_and_label_fn(filepath,
                self.label_paths[index], self.feature_fn, **self.kwargs)
            return features_and_labels
        elif self.include_labels:
            return ( (self.feature_fn(filepath, **self.feature_fn_args),
                self.get_label(index)) )
        else:
            # Tuple with None for label
            return ( self.feature_fn(filepath, **self.feature_fn_args), None )

class OneFileDataset(torch.utils.data.Dataset):
    """ A class for handling datasets with many examples per file."""

    def __init__(self, filepath, load_fn, feature_and_label_fn,
        start_index=None, end_index=None, **kwargs):

        # Validate args
        if bool(start_index is None) ^ bool(end_index is None):
            raise Exception("Need both or none of start_index and end_index")

        self.filepath=filepath
        self.load_fn=load_fn
        self.feature_and_label_fn=feature_and_label_fn
        self.kwargs = kwargs or {}

        self.data = load_fn(filepath)

        if start_index is not None:
            self.data = self.data[start_index:end_index]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.feature_and_label_fn(self.data[index], **self.kwargs)

class SwitchBoardLaughterDataset(torch.utils.data.Dataset):
    def __init__(self, df, audios_hash, feature_fn, sr, batch_size, subsample=True):
        # For training, we should set subsample to True, for val/testing, set to false
        # When subsample is False, we use the data in 'subsampled_offset/duration' every time
        # When it's True, we re-subsample to get more variation in time

        self.df = df
        self.audios_hash = audios_hash
        self.batch_size = batch_size
        self.subsample=subsample

        # Columns: [region start, region duration, subsampled region start, subsampled region duration, audio path, label]
        #self.df = pd.read_csv(data_file,sep='\t',header=None,
        #    names=['offset','duration','subsampled_offset','subsampled_duration','audio_path','label'])
        self.feature_fn = feature_fn
        self.sr = sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        audio_file = self.audios_hash[self.df.audio_path[index]]
        
        if self.subsample:
            audio_file_length = librosa.core.samples_to_time(len(audio_file),sr=self.sr)
            offset, duration = audio_utils.subsample_time(self.df.offset[index], self.df.duration[index], audio_file_length=audio_file_length,
                subsample_length=1.0, padding_length=0.5)
        else:
            offset = self.df.subsampled_offset[index]
            duration = self.df.subsampled_duration[index]

        X = self.feature_fn(y=audio_file, sr=self.sr, offset=offset, duration=duration)
        y = self.df.label[index]
        return (X,y)

class SwitchBoardLaughterInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, feature_fn, sr=8000, n_frames=44):
        self.audio_path = audio_path
        self.n_frames = n_frames
        self.feature_fn = feature_fn
        self.sr = sr
        self.n_frames = n_frames

        self.y, _ = librosa.load(audio_path, sr=sr)
        self.features = feature_fn(y=self.y,sr=self.sr)

    def __len__(self):
        return len(self.features)-self.n_frames

    def __getitem__(self, index):
        # return None for labels
        return (self.features[index:index+self.n_frames], None)

class AudioBatchDataset(torch.utils.data.Dataset):
    """ A class to load batches of audio files in Torch
        This will just return a list of audio files and labels
        Example Usage:
        1. train_dataset = AudioBatchDataset(
            train_audio_files, train_labels, batch_size=32, n_processes=10)
        2. training_generator = torch.utils.data.DataLoader(train_dataset)
        3. for signals, labels in training_generator:
            ... run training code ...

    """
    def __init__(self, filepaths, labels, batch_size=32, n_processes=1):
        self.labels = labels
        self.filepaths = filepaths
        self.batch_size = batch_size
        self.n_processes = n_processes
    def __len__(self):
        # The length as the number of batches
        return int(np.ceil(float(len(self.filepaths)) / self.batch_size))
    def __getitem__(self, index):
        # Get a batch
        files = self.filepaths[index:index+batch_size]
        signals = parallel_load_audio_batch(files, self.n_processes)
        labels = self.labels[index:index+batch_size]
        return signals, labels


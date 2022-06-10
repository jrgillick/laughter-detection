import librosa
import numpy as np
import torch
import torch.utils.data


class SwitchBoardLaughterInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, wav, feature_fn, orig_sr, target_sr=8000, n_frames=44):
        self.n_frames = n_frames
        self.feature_fn = feature_fn
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.n_frames = n_frames
        self.y = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
        self.features = feature_fn(y=self.y, sr=self.target_sr)

    def __len__(self):
        return len(self.features) - self.n_frames

    def __getitem__(self, index):
        # return None for labels
        return (self.features[index:index + self.n_frames], None)

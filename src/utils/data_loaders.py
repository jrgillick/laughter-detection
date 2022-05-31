import librosa
import numpy as np
import torch
import torch.utils.data


class SwitchBoardLaughterInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, audio_path, feature_fn, sr=8000, n_frames=44):
        self.audio_path = audio_path
        self.n_frames = n_frames
        self.feature_fn = feature_fn
        self.sr = sr
        self.n_frames = n_frames

        self.y, _ = librosa.load(audio_path, sr=sr)
        self.features = feature_fn(y=self.y, sr=self.sr)

    def __len__(self):
        return len(self.features) - self.n_frames

    def __getitem__(self, index):
        # return None for labels
        return (self.features[index:index + self.n_frames], None)

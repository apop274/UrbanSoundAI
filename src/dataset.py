"""
Dataset utilities: load UrbanSound8K metadata, filter classes,
load audio with librosa, and convert to mel spectrograms.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import librosa

from . import config


def load_and_filter_metadata(csv_path=None):
    """
    Load UrbanSound8K CSV and filter to TARGET_CLASSES only.
    Returns a DataFrame with columns including 'slice_file_name', 'fold', 'class'.
    """
    path = csv_path or config.METADATA_CSV
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    df = pd.read_csv(path)
    df = df[df["class"].isin(config.TARGET_CLASSES)].reset_index(drop=True)
    return df


def get_audio_path(row):
    """Given a metadata row with 'fold' and 'slice_file_name', return full path."""
    fold = row["fold"]
    fname = row["slice_file_name"]
    return os.path.join(config.AUDIO_DIR, f"fold{fold}", fname)


def audio_to_mel_spectrogram(
    file_path,
    sr=config.SR,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
    n_time_steps=config.SPEC_TIME_STEPS,
):
    """
    Load a WAV file and convert it to a fixed-size mel spectrogram.
    Returns numpy array of shape (n_mels, n_time_steps). Trims or pads on time axis.
    """
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    # Log scale (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # (n_mels, time) -> ensure time dimension is n_time_steps
    _, T = mel_db.shape
    if T >= n_time_steps:
        mel_db = mel_db[:, :n_time_steps]
    else:
        pad = np.zeros((n_mels, n_time_steps - T), dtype=mel_db.dtype)
        mel_db = np.concatenate([mel_db, pad], axis=1)
    return mel_db.astype(np.float32)


class UrbanSoundSpectrogramDataset(Dataset):
    """
    PyTorch Dataset that yields (mel_spectrogram, class_index) for UrbanSound8K
    rows filtered to TARGET_CLASSES.
    """

    def __init__(self, metadata_df, transform=None):
        self.metadata = metadata_df
        self.class_to_idx = {c: i for i, c in enumerate(config.TARGET_CLASSES)}
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        path = get_audio_path(row)
        spec = audio_to_mel_spectrogram(path)
        # Add channel dim for CNN: (1, n_mels, time)
        spec = spec[np.newaxis, :, :]
        label = self.class_to_idx[row["class"]]
        if self.transform:
            spec = self.transform(spec)
        return torch.from_numpy(spec), label

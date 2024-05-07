# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:34:35 2024

@author: Denebola Biswas
"""
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
import os
import torchaudio.transforms as T
from functools import lru_cache
from typing import Union
import matplotlib as plt
import librosa

import numpy as np
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "src\\models\\assets\\mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def log_mel_spectrogram(audio: torch.Tensor, n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    """
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[:, :-1].abs() ** 2
    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    """
    for i in range(len(audio_paths)):
        waveform, sample_rate = torchaudio.load(audio_paths[i], normalize=True)
        transform = T.MelSpectrogram(sample_rate)
        mel_specgram = transform(waveform)  # (channel, n_mels, time)
        plot_spectrogram(mel_specgram)

directory = 'D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild\\release_in_the_wild'
file_names = os.listdir(directory)
wav_files = [file for file in file_names if file.endswith('.flac')]
audio_paths = [os.path.join(directory, file) for file in wav_files]
log_mel_spectrogram(audio_paths,N_MELS)

# Step 4: Initialize DataLoader 
batch_size = 8
dataloader = DataLoader(audio_paths, batch_size=batch_size, shuffle=False)
print(dataloader)


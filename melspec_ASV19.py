# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 01:18:58 2024

@author: Denebola Biswas
"""

import numpy as np
import soundfile as sf
from IPython.display import Audio
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display

dir_path_LA_dev= "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\flac\\"


def save_spectogram_1(file_name, label):
    file_name=file_name.strip()+'.flac'
    clip, sample_rate = sf.read(dir_path_LA_dev+file_name)
    file_name = file_name.split('/')[0]
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\spectrograms\\'+label+'\\'+file_name.replace(
        '.flac', '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')


text_LA_dev= pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\LA_dev_meta.csv")

for file in range(len(text_LA_dev["clip"])):
    save_spectogram_1(text_LA_dev["clip"][file], text_LA_dev["label"][file])

print("yayayay")

dir_path_LA_eval= "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\flac\\"


def save_spectogram_2(file_name, label):
    file_name=file_name.strip()+'.flac'
    clip, sample_rate = sf.read(dir_path_LA_eval+file_name)
    file_name = file_name.split('/')[0]
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\'+label+'\\'+file_name.replace(
        '.flac', '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')


text_LA_eval= pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\LA_eval_meta.csv")

for file in range(len(text_LA_eval["clip"])):
    save_spectogram_2(text_LA_eval["clip"][file], text_LA_eval["label"][file])

print("yayayay2")

dir_path_LA_train= "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\flac\\"


def save_spectogram_3(file_name, label):
    file_name=file_name.strip()+'.flac'
    print(dir_path_LA_train+file_name)
    clip, sample_rate = sf.read(dir_path_LA_train+file_name)
    file_name = file_name.split('/')[0]
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\spectrograms\\'+label+'\\'+file_name.replace(
        '.flac', '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')


text_LA_train= pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\LA_train_meta.csv")

for file in range(len(text_LA_train["clip"])):
    file_name=text_LA_train["clip"][file]
    print(dir_path_LA_train + file_name)
    save_spectogram_3(text_LA_train["clip"][file], text_LA_train["label"][file])

print("yayayay3")




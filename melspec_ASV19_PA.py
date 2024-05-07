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

dir_path_PA_dev= "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\flac\\"


def save_spectogram_1(file_name, label):
    file_name=file_name.strip()+'.flac'
    clip, sample_rate = sf.read(dir_path_PA_dev+file_name)
    file_name = file_name.split('/')[0]
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'D:\\vit btech final year 2023\Capstone\Datasets\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\spectrograms\\'+label+'\\'+file_name.replace(
        '.flac', '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')


text_PA_dev= pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\PA_dev_meta.csv")

for file in range(len(text_PA_dev["clip"])):
    save_spectogram_1(text_PA_dev["clip"][file], text_PA_dev["label"][file])

print("yayayay")

dir_path_PA_eval= "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\flac\\"


def save_spectogram_2(file_name, label):
    file_name=file_name.strip()+'.flac'
    clip, sample_rate = sf.read(dir_path_PA_eval+file_name)
    file_name = file_name.split('/')[0]
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\spectrograms\\'+label+'\\'+file_name.replace(
        '.flac', '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')


text_PA_eval= pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\PA_eval_meta.csv")

for file in range(len(text_PA_eval["clip"])):
    save_spectogram_2(text_PA_eval["clip"][file], text_PA_eval["label"][file])

print("yayayay2")

dir_path_PA_train= "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\flac\\"


def save_spectogram_3(file_name, label):
    file_name=file_name.strip()+'.flac'
    clip, sample_rate = sf.read(dir_path_PA_train+file_name)
    file_name = file_name.split('/')[0]
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\spectrograms\\'+label+'\\'+file_name.replace(
        '.flac', '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')


text_PA_train= pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\PA_train_meta.csv")

for file in range(len(text_PA_train["clip"])):
    save_spectogram_3(text_PA_train["clip"][file], text_PA_train["label"][file])

print("yayayay3")




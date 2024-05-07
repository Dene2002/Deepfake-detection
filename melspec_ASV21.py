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

dir_path= "D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\ASV_flacaudio_full\\"


def save_spectogram(file_name, label):
    file_name=file_name.strip()+'.flac'
    print(file_name)
    clip, sample_rate = sf.read(dir_path+file_name)
    file_name = file_name.split('/')[0]
    fig = plt.figure(figsize=[0.72, 0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = 'D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\spectrograms\\'+label.strip()+'\\'+file_name.replace(
        '.flac', '.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close('all')


text = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\keys\\CM\\ASV21_meta.csv")
print(len(text))

for file in range(293279,len(text["clip"])):
    save_spectogram(text["clip"][file], text["label"][file])

print("yayayay")



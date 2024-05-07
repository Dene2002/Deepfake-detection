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

dir_path="D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"

def save_spectogram(file_name,label):
    clip, sample_rate = sf.read(dir_path+'\\'+file_name)
    sample_rate = sf.info.samplerate
    print("Sample rate:", sample_rate)
    file_name = file_name.split('/')[0]
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = 'D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms\\'+label+'\\'+file_name.replace('.wav','.png')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close('all')
    
    
text = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild\\meta.csv")
print(text)
for file in range(len(text["file"])):
    save_spectogram(text["file"][file],text["label"][file])

import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import soundfile as sf
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
import os
# import matplotlib.pyplot as plt
# import librosa.display
from keras.layers import Layer,MaxPooling2D

from keras.models import load_model
import json

input_size = (223, 221)
model = load_model('output_mesonet.h5')
dicti={}

def ASV21():
    print("testing for ASV spoof 2021")
    test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\keys\\CM\\ASV21_MESO_meta.csv")
    test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\spectrograms\\bonafide_new"
    test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\spectrograms\\spoof_new"
    dir_list1 = os.listdir(test_set1)
    dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
    dir_list2 = os.listdir(test_set2)
    dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
    dir_list_final = dir_list1+dir_list2
    print(dir_list_final)
    for filename in dir_list_final:
        mel_spec=filename
        if filename in dir_list_final:
            print(filename)
            img = image.load_img(mel_spec,target_size=(223, 221))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array=img_array/255.0
            predicted_class = model.predict(img_array)
            # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
            predicted_class = np.argmax(predicted_class)
            # print("Predicted Class:", predicted_class)
            filename = os.path.splitext(os.path.basename(filename))[0]
            print(filename)
            with open("MESO_ASV21.txt", "a") as file:
                # Append data to the file
                file.write(f"{filename}: {predicted_class}\n")
    #     if filename in test_csv.values:
    #         # Find the row index where the file name is present in the Excel sheet
    #         row_index = test_csv.index[test_csv['file'] == filename[:-4] + ".flac"].tolist()[0]
    #         # Update the 4th column of that row to "run"
    #         test_csv.iloc[row_index,4 ] = predicted_class
    #         test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild\\meta.csv", index=False)
    #
    # predicted_labels = test_csv['PL_MS_LCNN'].tolist()
    # true_labels = test_csv['tlabel'].tolist()
    #
    # conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #
    # # Print or visualize the confusion matrix
    # print("Confusion Matrix for ASV21 of LCNN:")
    # print(conf_matrix)
    #
    # test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
    #
    # print("accuracy for ASV21 of LCNN: ",test_accuracy)


ASV21()

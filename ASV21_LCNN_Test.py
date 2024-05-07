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
from keras.layers import Layer, MaxPooling2D
import json


class MaxFeatureMap2D(Layer):
    """Max feature map (along 2D)"""

    def __init__(self, max_dim=1, **kwargs):
        super(MaxFeatureMap2D, self).__init__(**kwargs)
        self.max_dim = max_dim

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        if self.max_dim != 1:
            raise ValueError("MaxFeatureMap: The default value for max_dim should be 1.")

        # Perform max pooling along the channel dimension
        x = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(inputs)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] // 2, input_shape[3])

    def get_config(self):
        config = super().get_config()
        config.update({'max_dim': self.max_dim})
        return config


#Testing data
input_size = (223, 221, 3)
model = load_model('output_lcnn.h5', custom_objects={'MaxFeatureMap2D': MaxFeatureMap2D})
dicti = {}


# def ASV21():
print("testing for ASV spoof 2021")
test_csv = pd.read_csv(
    "D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\keys\\CM\\ASV21_LCNN_meta.csv")
test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\spectrograms\\bonafide_new"
test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\spectrograms\\spoof_new"
dir_list1 = os.listdir(test_set1)
dir_list1 = [test_set1 + "\\" + files for files in dir_list1]
dir_list2 = os.listdir(test_set2)
dir_list2 = [test_set2 + "\\" + files for files in dir_list2]
dir_list_final = dir_list1 + dir_list2
print(len(dir_list_final))
print(test_csv.values)
print(test_csv.__len__())
for filename in dir_list_final:
    print("for loop")
    mel_spec = filename
    # filename = os.path.splitext(os.path.basename(filename))[0]
    print(filename)
    if filename in dir_list_final:
        print("Hy")
        img = image.load_img(mel_spec, target_size=(223, 221))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predicted_class = model.predict(img_array)
        # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
        predicted_class = np.argmax(predicted_class)
        # print("Predicted Class:", predicted_class)
        filename = os.path.splitext(os.path.basename(filename))[0]
        print(filename)
        with open("LCNN_ASV21.txt", "a") as file:
            # Append data to the file
            file.write(f"{filename}: {predicted_class}\n")


#     #     if filename in test_csv.values:
#     #         # Find the row index where the file name is present in the Excel sheet
#     #         row_index = test_csv.index[test_csv['file'] == filename[:-4] + ".flac"].tolist()[0]
#     #         # Update the 4th column of that row to "run"
#     #         test_csv.iloc[row_index,4 ] = predicted_class
#     #         test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild\\meta.csv", index=False)
#     #
#     # predicted_labels = test_csv['PL_MS_LCNN'].tolist()
#     # true_labels = test_csv['tlabel'].tolist()
#     #
#     # conf_matrix = confusion_matrix(true_labels, predicted_labels)
#     #
#     # # Print or visualize the confusion matrix
#     # print("Confusion Matrix for ASV21 of LCNN:")
#     # print(conf_matrix)
#     #
#     # test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#     #
#     # print("accuracy for ASV21 of LCNN: ",test_accuracy)
#
#
# ASV21()

# import csv
#
# def select_rows(excel_file, sheet_name, start_row, end_row):
#     selected_rows = []
#     with open(excel_file, 'r') as file:
#         csv_reader = csv.reader(file)
#         for row_num, row in enumerate(csv_reader, start=1):
#             if start_row <= row_num <= end_row:
#                 selected_rows.append(row)
#     output_file='ASV21_LCNN_meta.csv'
#     with open(output_file, 'w', newline='') as file:
#         csv_writer = csv.writer(file)
#         csv_writer.writerows(selected_rows)
#
# # Example usage:
# select_rows('D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\keys\\CM\\ASV21_meta.csv', 'Sheet1', 481573, 521567)  # Selects rows 2 to 5
#

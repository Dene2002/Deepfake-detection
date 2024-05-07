import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
import os
# import matplotlib.pyplot as plt
# import librosa.display
from keras.layers import Layer,MaxPooling2D
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
dicti={}
dicti1={}

# def IntheWild():
#     print("testing for In the Wild Dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild\\meta.csv")
#     # test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms\\spoof"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms\\bona-fide"
#     dir_list1 = os.listdir(test_set1)
#     dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
#     dir_list2 = os.listdir(test_set2)
#     dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
#     dir_list_final = dir_list1+dir_list2
#     print(dir_list_final)
#     for filename in dir_list_final:
#         print("file name: "+filename)
#         img = image.load_img(filename,target_size=(223, 221))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array=img_array/255.0
#         predicted_class = model.predict(img_array)
#         # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
#         predicted_class = np.argmax(predicted_class)
#         # print("Predicted Class:", predicted_class)
#         filename=os.path.splitext(os.path.basename(filename))[0]+".wav"
#         print(predicted_class)
#         if filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['file'] == filename[:-4] + ".wav"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,4 ] = predicted_class
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild\\meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_LCNN'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for ITW of LCNN:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for ITW of LCNN: ",test_accuracy)

# def ASVSpoof2019_LA_dev():
#     print("LA_development_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\LA_dev_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\spectrograms\\bonafide"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\spectrograms\\spoof"
#     dir_list1 = os.listdir(test_set1)
#     dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
#     dir_list2 = os.listdir(test_set2)
#     dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
#     dir_list_final = dir_list1+dir_list2
#     print(dir_list_final)
#     for filename in dir_list_final:
#         print("file name: "+filename)
#         img = image.load_img(filename,target_size=(223, 221))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array=img_array/255.0
#         predicted_class = model.predict(img_array)
#         # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
#         predicted_class = np.argmax(predicted_class)
#         # print("Predicted Class:", predicted_class)
#         filename=os.path.splitext(os.path.basename(filename))[0]+".flac"
#         print(predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,4 ] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\LA_dev_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_LCNN'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for LA_dev of LCNN:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for LA_dev of LCNN: ",test_accuracy)
#


def ASVSpoof2019_LA_eval():
    print("LA_evaluation_dataset")
    test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\LA_eval_meta_LCNN.csv")
    #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
    test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\bonafide"
    test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\spoof"
    dir_list1 = os.listdir(test_set1)
    dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
    dir_list2 = os.listdir(test_set2)
    dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
    dir_list_final = dir_list1+dir_list2
    print(dir_list_final)
    for filename in dir_list_final:
        print("file name: "+filename)
        img = image.load_img(filename,target_size=(223, 221))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array=img_array/255.0
        predicted_class = model.predict(img_array)
        # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
        predicted_class = np.argmax(predicted_class)
        # print("Predicted Class:", predicted_class)
        filename=os.path.splitext(os.path.basename(filename))[0]+".flac"
        dicti[filename]=predicted_class
        print(dicti)
    with open('LCNN_LA19.json', 'w') as f:
        json.dump(dicti, f)
    #     print(predicted_class)
    #     print(filename)
    #     if  filename in test_csv.values:
    #         # Find the row index where the file name is present in the Excel sheet
    #         row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
    #         # Update the 4th column of that row to "run"
    #         test_csv.iloc[row_index,4 ] = predicted_class
    #         # test_csv.iloc[index,3]=predicted_class
    #         # index+=1
    #         test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\LA_eval_meta.csv", index=False)
    #
    # predicted_labels = test_csv['PL_MS_LCNN'].tolist()
    # true_labels = test_csv['tlabel'].tolist()
    #
    # conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #
    # # Print or visualize the confusion matrix
    # print("Confusion Matrix for LA_eval of LCNN:")
    # print(conf_matrix)
    #
    # test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
    #
    # print("accuracy for LA_eval of LCNN: ",test_accuracy)



# def ASVSpoof2019_LA_train():
#     print("LA_training_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\LA_train_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\spectrograms\\bonafide"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\spectrograms\\spoof"
#     dir_list1 = os.listdir(test_set1)
#     dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
#     dir_list2 = os.listdir(test_set2)
#     dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
#     dir_list_final = dir_list1+dir_list2
#     print(dir_list_final)
#     for filename in dir_list_final:
#         print("file name: "+filename)
#         img = image.load_img(filename,target_size=(223, 221))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array=img_array/255.0
#         predicted_class = model.predict(img_array)
#         # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
#         predicted_class = np.argmax(predicted_class)
#         # print("Predicted Class:", predicted_class)
#         filename=os.path.splitext(os.path.basename(filename))[0]+".flac"
#         print(predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,4 ] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\LA_train_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_LCNN'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for LA_train of LCNN:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for LA_train of LCNN: ",test_accuracy)
#


# def ASVSpoof2019_PA_dev():
#     print("PA_development_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\PA_dev_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\spectrograms\\bonafide"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\spectrograms\\spoof"
#     dir_list1 = os.listdir(test_set1)
#     dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
#     dir_list2 = os.listdir(test_set2)
#     dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
#     dir_list_final = dir_list1+dir_list2
#     print(dir_list_final)
#     for filename in dir_list_final:
#         print("file name: "+filename)
#         img = image.load_img(filename,target_size=(223, 221))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array=img_array/255.0
#         predicted_class = model.predict(img_array)
#         # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
#         predicted_class = np.argmax(predicted_class)
#         # print("Predicted Class:", predicted_class)
#         filename=os.path.splitext(os.path.basename(filename))[0]+".flac"
#         print(predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,4 ] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\PA_dev_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_LCNN'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for PA_dev of LCNN:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for PA_dev of LCNN: ",test_accuracy)
#



    #     json.dump(dicti, f)def ASVSpoof2019_PA_eval():
    # print("PA_evaluation_dataset")
    # test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\PA_eval_meta_LCNN.csv")
    # #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
    # test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\spectrograms\\bonafide"
    # test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\spectrograms\\spoof"
    # dir_list1 = os.listdir(test_set1)
    # dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
    # dir_list2 = os.listdir(test_set2)
    # dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
    # dir_list_final = dir_list1+dir_list2
    # print(dir_list_final)
    # for filename in dir_list_final:
    #     print("file name: "+filename)
    #     img = image.load_img(filename,target_size=(223, 221))
    #     img_array = image.img_to_array(img)
    #     img_array = np.expand_dims(img_array, axis=0)
    #     img_array=img_array/255.0
    #     predicted_class = model.predict(img_array)
    #     # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
    #     predicted_class = np.argmax(predicted_class)
    #     # print("Predicted Class:", predicted_class)
    #     filename=os.path.splitext(os.path.basename(filename))[0]+".flac"
    #     print(filename)
    #     print(predicted_class)
    #     dicti1[filename]=predicted_class
    # print(dicti)
    # with open('LCNN_PA.json', 'w') as f
    #     if  filename in test_csv.values:
    #         # Find the row index where the file name is present in the Excel sheet
    #         row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
    #         # Update the 4th column of that row to "run"
    #         test_csv.iloc[row_index,4 ] = predicted_class
    #         # test_csv.iloc[index,3]=predicted_class
    #         # index+=1
    #         test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\PA_eval_meta.csv", index=False)
    #
    # predicted_labels = test_csv['PL_MS_LCNN'].tolist()
    # true_labels = test_csv['tlabel'].tolist()
    #
    # conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #
    # # Print or visualize the confusion matrix
    # print("Confusion Matrix for PA_eval of LCNN:")
    # print(conf_matrix)
    #
    # test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
    #
    # print("accuracy for PA_eval of LCNN: ",test_accuracy)



# def ASVSpoof2019_PA_train():
#     print("PA_training_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\PA_train_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\spectrograms\\bonafide"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\spectrograms\\spoof"
#     dir_list1 = os.listdir(test_set1)
#     dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
#     dir_list2 = os.listdir(test_set2)
#     dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
#     dir_list_final = dir_list1+dir_list2
#     print(dir_list_final)
#     for filename in dir_list_final:
#         print("file name: "+filename)
#         img = image.load_img(filename,target_size=(223, 221))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array=img_array/255.0
#         predicted_class = model.predict(img_array)
#         # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
#         predicted_class = np.argmax(predicted_class)
#         # print("Predicted Class:", predicted_class)
#         filename=os.path.splitext(os.path.basename(filename))[0]+".flac"
#         print(predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,4 ] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\PA_train_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_LCNN'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for PA_train of LCNN:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for PA_train of LCNN: ",test_accuracy)



ASVSpoof2019_LA_eval()









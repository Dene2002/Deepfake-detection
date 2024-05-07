from keras.models import load_model

input_size = (223, 221)
model = load_model('output_mesonet.h5')
dicti = {}
# test_csv = pd.read_excel("D:\\vit btech final year 2023\\Capstone\\whisper_code\\Sample_Dataset.xlsx")
# # index = 0
# test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
# # test_set1 = "D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms_test\\bona-fide"
# # test_set2 = "D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms_test\\spoof"
# # dir_list1 = os.listdir(test_set1)
# # dir_list1 = [test_set1+"\\"+ files for files in dir_list1]
# # dir_list2 = os.listdir(test_set2)
# # dir_list2 = [test_set2+"\\"+ files for files in dir_list2]
# # dir_list_final = dir_list1+dir_list2
# # print(dir_list_final)
# # for filename in dir_list_final:
# #     print("file name: "+filename)
# for file in range(len(test_csv["file"])):
#     sound_file = test_csv["file"][file]
#     label = test_csv["label"][file]
#     sound_file = sound_file.split('/')[0]
#     clip, sample_rate = sf.read(test_dir + '\\' + sound_file)
#     fig = plt.figure(figsize=[0.72, 0.72])
#     ax = fig.add_subplot(111)
#     ax.axes.get_xaxis().set_visible(False)
#     ax.axes.get_yaxis().set_visible(False)
#     ax.set_frame_on(False)
#     S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
#     librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
#     filename = 'D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms_test\\' + label + '\\' + sound_file.replace(
#         '.wav', '.png')
#     plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
#     plt.close('all')
#     img = image.load_img(filename,target_size=(223, 221,3))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array=img_array/255.0
#     predicted_class = model.predict(img_array)
#     # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
#     predicted_class = np.argmax(predicted_class)
#     # print("Predicted Class:", predicted_class)
#     filename=os.path.splitext(os.path.basename(filename))[0]+".wav"
#     print(predicted_class)
#     if  filename in test_csv.values:
#         # Find the row index where the file name is present in the Excel sheet
#         row_index = test_csv.index[test_csv['file'] == filename[:-4] + ".wav"].tolist()[0]
#         # Update the 4th column of that row to "run"
#         test_csv.iloc[row_index,5 ] = predicted_class
#         # test_csv.iloc[index,3]=predicted_class
#         # index+=1
#         test_csv.to_excel("D:\\vit btech final year 2023\\Capstone\\whisper_code\\Sample_Dataset.xlsx", index=False)
#
# predicted_labels = test_csv['PL_MS_MESO'].tolist()
# true_labels = test_csv['tlabel'].tolist()
#
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
# # Print or visualize the confusion matrix
# print("Confusion Matrix for mesonet of mel spectrogram:")
# print(conf_matrix)
#
# test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
# print(test_accuracy)

import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
import os
import json


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
#             test_csv.iloc[row_index,5 ] = predicted_class
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild\\meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_MESO'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for ITW of Mesonet:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for ITW of Mesonet: ",test_accuracy)
#
#

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
#             test_csv.iloc[row_index,5 ] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\LA_dev_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_MESO'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for LA_dev of MESO:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for LA_dev of MESO: ",test_accuracy)


def ASVSpoof2019_LA_eval():
    print("LA_evaluation_dataset")
    test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\bonafide_new"
    test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\spoof_new"
    dir_list1 = os.listdir(test_set1)
    dir_list1 = [test_set1 + "\\" + files for files in dir_list1]
    dir_list2 = os.listdir(test_set2)
    dir_list2 = [test_set2 + "\\" + files for files in dir_list2]
    dir_list_final = dir_list1 + dir_list2
    print(dir_list_final)
    for filename in dir_list_final:
        print("file name: " + filename)
        img = image.load_img(filename, target_size=(223, 221))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predicted_class = model.predict(img_array)
        # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
        predicted_class = np.argmax(predicted_class)
        # print("Predicted Class:", predicted_class)
        filename = os.path.splitext(os.path.basename(filename))[0]
        with open("MESO_ASV19.txt", "a") as file:
            # Append data to the file
            file.write(f"{filename}: {predicted_class}\n")
    #     if  filename in test_csv.values:
    #         # Find the row index where the file name is present in the Excel sheet
    #         row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
    #         # Update the 4th column of that row to "run"
    #         test_csv.iloc[row_index,5 ] = predicted_class
    #         # test_csv.iloc[index,3]=predicted_class
    #         # index+=1
    #         test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\LA_eval_meta.csv", index=False)
    #
    # predicted_labels = test_csv['PL_MS_MESO'].tolist()
    # true_labels = test_csv['tlabel'].tolist()
    #
    # conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #
    # # Print or visualize the confusion matrix
    # print("Confusion Matrix for LA_eval of MESO:")
    # print(conf_matrix)
    #
    # test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
    #
    # print("accuracy for LA_eval of MESO: ",test_accuracy)


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
#             test_csv.iloc[row_index,5] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\LA_train_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_MESO'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for LA_train of MESO:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for LA_train of MESO: ",test_accuracy)
#
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
#             test_csv.iloc[row_index,5] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\PA_dev_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_MESO'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for PA_dev of MESO:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for PA_dev of MESO: ",test_accuracy)


# def ASVSpoof2019_PA_eval():
#     print("PA_evaluation_dataset")
#     test_csv = pd.read_csv(
#         "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\PA_eval_meta_MESO.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\spectrograms\\bonafide"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\spectrograms\\spoof"
#     dir_list1 = os.listdir(test_set1)
#     dir_list1 = [test_set1 + "\\" + files for files in dir_list1]
#     dir_list2 = os.listdir(test_set2)
#     dir_list2 = [test_set2 + "\\" + files for files in dir_list2]
#     dir_list_final = dir_list1 + dir_list2
#     print(dir_list_final)
#     for filename in dir_list_final:
#         print("file name: " + filename)
#         img = image.load_img(filename, target_size=(223, 221))
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array = img_array / 255.0
#         predicted_class = model.predict(img_array)
#         # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
#         predicted_class = np.argmax(predicted_class)
#         filename = os.path.splitext(os.path.basename(filename))[0] + ".flac"
#         dicti1 [filename] = predicted_class
#     print(dicti1)
#     with open('MESO_PA.json', 'w') as f:
#         json.dump(dicti1, f)
    #     if  filename in test_csv.values:
    #         # Find the row index where the file name is present in the Excel sheet
    #         row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
    #         # Update the 4th column of that row to "run"
    #         test_csv.iloc[row_index,5] = predicted_class
    #         # test_csv.iloc[index,3]=predicted_class
    #         # index+=1
    #         test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\PA_eval_meta.csv", index=False)
    #
    # predicted_labels = test_csv['PL_MS_MESO'].tolist()
    # true_labels = test_csv['tlabel'].tolist()
    #
    # conf_matrix = confusion_matrix(true_labels, predicted_labels)
    #
    # # Print or visualize the confusion matrix
    # print("Confusion Matrix for PA_eval of MESO:")
    # print(conf_matrix)
    #
    # test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
    #
    # print("accuracy for PA_eval of MESO: ",test_accuracy)


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
#             test_csv.iloc[row_index,5] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\PA_train_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_MESO'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for PA_train of MESO:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for PA_train of MESO: ",test_accuracy)
#


ASVSpoof2019_LA_eval()

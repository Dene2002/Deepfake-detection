import os
from keras.src.saving.object_registration import register_keras_serializable
import joblib
from tensorflow.keras.preprocessing import image

import tensorflow as tf
from keras.layers import Concatenate
from keras.src.saving.object_registration import register_keras_serializable

from tensorflow.keras import layers, Model

import numpy as np




NUM_CLASSES = 2
input_size = (223, 221, 3)


class PreEmphasis(layers.Layer):

    def __init__(self, coef=0.97):
        super(PreEmphasis, self).__init__()

        self.coef = coef

    def call(self, inputs):
        print("PreEmphasis inputs shape:", inputs.shape)

        result = inputs - self.coef * tf.pad(inputs, [[0, 0], [1, 0], [1, 0], [0, 0]], mode='REFLECT')[:, :-1, :-1, :]

        print("PreEmphasis output shape:", result.shape)

        return result


class Bottle2neck(layers.Layer):
    def __init__(self, inplanes, planes, dilation=None, scale=4, pool=False):
        super(Bottle2neck, self).__init__()
        self.inplanes = inplanes
        width = planes // scale
        self.conv1 = layers.Conv2D(width * scale, kernel_size=1)
        self.bn1 = layers.BatchNormalization()
        self.nums = scale - 1
        self.convs = [layers.Conv2D(width, kernel_size=3, dilation_rate=dilation, padding='same') for _ in range(self.nums)]
        self.bns = [layers.BatchNormalization() for _ in range(self.nums)]
        self.conv3 = layers.Conv2D(planes, kernel_size=1)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.width = width
        self.mp = layers.MaxPool2D(pool) if pool else None

        if inplanes != planes:
            self.residual = layers.Conv2D(planes, kernel_size=1, strides=1, use_bias=False)
        else:
            self.residual = layers.Lambda(lambda x: x)



class AlphaFeatureMapScaling(layers.Layer):
    def __init__(self, nb_dim):
        super(AlphaFeatureMapScaling, self).__init__()
        self.alpha = self.add_weight(shape=(1, 1, nb_dim), initializer='ones', trainable=True)  # Adjusted shape
        self.fc = layers.Dense(nb_dim)
        self.sig = layers.Activation('softmax')

    def call(self, x):
        y = tf.reduce_mean(x, axis=[1, 2], keepdims=True)  # Reduce along spatial dimensions
        y = self.sig(self.fc(y))
        x = x + self.alpha
        x = x * y
        return x

@register_keras_serializable()
class RawNet3(Model):

    def __init__(self, input_shape, model_scale=8, context=True, summed=True, encoder_type="ECA", nOut=1, out_bn=False,

                 sinc_stride=10, log_sinc=True, norm_sinc="mean"):

        super(RawNet3, self).__init__()

        self.context = context

        self.encoder_type = encoder_type

        self.log_sinc = log_sinc

        self.norm_sinc = norm_sinc

        self.out_bn = out_bn

        self.summed = summed

        self.preprocess = tf.keras.Sequential([

            PreEmphasis(),

            layers.LayerNormalization(epsilon=1e-4)

        ])

        self.conv1 = layers.Conv2D(1024 // 4, kernel_size=(251, 1), strides=(sinc_stride, 1), padding='same')

        self.relu = layers.ReLU()

        self.bn1 = layers.BatchNormalization()

        self.layer1 = Bottle2neck(1024 // 4, 1024, dilation=2, scale=model_scale, pool=(5, 5))

        self.layer2 = Bottle2neck(1024, 1024, dilation=3, scale=model_scale, pool=(3, 3))

        self.layer3 = Bottle2neck(1024, 1024, dilation=4, scale=model_scale)

        self.layer4 = layers.Conv2D(3 * 1024, kernel_size=1)

        self.layer5 = layers.BatchNormalization()

        self.pooling = layers.GlobalAveragePooling2D()

        self.fc1 = layers.Dense(256)

        self.fc2 = layers.Dense(NUM_CLASSES, activation="sigmoid")

        self.alpha_scaling = AlphaFeatureMapScaling(nb_dim=nOut)

    def call(self, inputs):

        if self.log_sinc:

            inputs = tf.math.log(tf.abs(inputs) + 1e-8)

        if self.norm_sinc == "mean":

            inputs = inputs - tf.math.reduce_mean(inputs, axis=-1, keepdims=True)

        elif self.norm_sinc == "cmn":

            inputs = inputs - tf.math.reduce_mean(inputs, axis=[1, 2], keepdims=True)

        x = self.preprocess(inputs)

        x = self.conv1(x)

        x = self.relu(x)

        x = self.bn1(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)

        x = self.pooling(x)

        x = self.fc1(x)

        if self.out_bn:

            x = self.alpha_scaling(x)

        x = self.fc2(x)

        return x

custom_objects = {
    'Rawnet3': RawNet3,
    'PreEmphasis': PreEmphasis,
    'Bottle2neck': Bottle2neck,
    'AlphaFeatureMapScaling':AlphaFeatureMapScaling
}

model = joblib.load("output_rawnet3.pkl")
dicti={}
test_set1 = "D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms_fast\\spoof"
test_set2 = "D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms_fast\\bona-fide"
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
    predicted_class = np.argmax(predicted_class)
    print("Predicted Class:", predicted_class)
    filename = os.path.splitext(os.path.basename(filename))[0] + ".wav"
    dicti[filename]=predicted_class
print(dicti)
    # print(predicted_class)
    # print(filename)
    # test_csv = pd.read_csv("D:\\whisper_code\\meta.csv")
    # if filename in test_csv.values:
    #     print(filename, "   fill ")
    #     # Find the row index where the file name is present in the Excel sheet
    #     row_index = test_csv.index[test_csv['file'] == filename[:-4] + ".wav"].tolist()[0]
    #     print("row_index", row_index)
    #     # Update the 4th column of that row to "run"
    #     test_csv.iloc[row_index, test_csv.columns.get_loc('PL_MS_SPECRNET')] = predicted_class
    #     print("predicted class: ", predicted_class)
    #     # test_csv.iloc[index,3]=predicted_class
    #     # index+=1
    #     test_csv.to_csv("D:\\whisper_code\\meta.csv", index=False)

# Calculate true positives, false positives, and ROC curve




# def ASVSpoof2019_LA_dev():
#     model = joblib.load("output_specRnet.pkl")
#     print("LA_development_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\LA_dev_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\spectrograms\\spoof"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\spectrograms\\bonafide"
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
#         print("predicted class: ",predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,7] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_dev\\LA_dev_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_SPECRNET'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for LA_dev of Specrnet:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for LA_dev of specrnet: ",test_accuracy)
#
#
#
# def ASVSpoof2019_LA_eval():
#     model = joblib.load("output_specRnet.pkl")
#     print("LA_evaluation_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\LA_eval_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\spoof"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\bonafide"
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
#         print("predicted class: ",predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,7] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\LA_eval_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_SPECRNET'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for LA_eval of specrnet:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for LA_eval of specrnet: ",test_accuracy)
#
#
#
# def ASVSpoof2019_LA_train():
#     model = joblib.load("output_specRnet.pkl")
#     print("LA_training_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\LA_train_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\spectrograms\\spoof"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\spectrograms\\bonafide"
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
#         print("predicted class: ",predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,7] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_train\\LA_train_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_SPECRNET'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for LA_train of specrnet:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for LA_train of Specrnet: ",test_accuracy)
#
#
# def ASVSpoof2019_PA_dev():
#     model = joblib.load("output_specRnet.pkl")
#     print("PA_development_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\PA_dev_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\spectrograms\\spoof"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\spectrograms\\bonafide"
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
#         print("predicted class: ",predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,7] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_dev\\PA_dev_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_SPECRNET'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for PA_dev of specrnet:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for PA_dev of specrnet: ",test_accuracy)
#
#
#
# def ASVSpoof2019_PA_eval():
#     model = joblib.load("output_specRnet.pkl")
#     print("PA_evaluation_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\PA_eval_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\spectrograms\\spoof"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\spectrograms\\bonafide"
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
#         print("predicted class: ",predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,7] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_eval\\PA_eval_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_SPECRNET'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for PA_eval of Specrnet:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for PA_eval of Specrnet: ",test_accuracy)
#
#
#
# def ASVSpoof2019_PA_train():
#     model = joblib.load("output_specRnet.pkl")
#     print("PA_training_dataset")
#     test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\PA_train_meta.csv")
#     #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
#     test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\spectrograms\\spoof"
#     test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\spectrograms\\bonafide"
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
#         print("predicted class: ",predicted_class)
#         print(filename)
#         if  filename in test_csv.values:
#             # Find the row index where the file name is present in the Excel sheet
#             row_index = test_csv.index[test_csv['clip'] == filename[:-4] + ".flac"].tolist()[0]
#             # Update the 4th column of that row to "run"
#             test_csv.iloc[row_index,7] = predicted_class
#             # test_csv.iloc[index,3]=predicted_class
#             # index+=1
#             test_csv.to_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\PA\\PA\\ASVspoof2019_PA_train\\PA_train_meta.csv", index=False)
#
#     predicted_labels = test_csv['PL_MS_SPECRNET'].tolist()
#     true_labels = test_csv['tlabel'].tolist()
#
#     conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
#     # Print or visualize the confusion matrix
#     print("Confusion Matrix for PA_train of Specrnet:")
#     print(conf_matrix)
#
#     test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
#     print("accuracy for PA_train of Specrnet: ",test_accuracy)
#
#
#
# print("Printing for ASV Spoof 2019 LA")
# ASVSpoof2019_LA_train()
# ASVSpoof2019_LA_dev()
# ASVSpoof2019_LA_eval()
# print("Printing for ASV Spoof 2019_PA")
# ASVSpoof2019_PA_dev()
# ASVSpoof2019_PA_train()
# ASVSpoof2019_PA_eval()
#
#
#
#
#
#

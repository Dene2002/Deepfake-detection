import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.generic_utils import register_keras_serializable
#from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# test_set1 = "D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms_fast\\spoof"
# test_set2 = "D:\\vit btech final year 2023\\Capstone\\whisper_code\\spectrograms_fast\\bona-fide"
# dir_list1 = os.listdir(test_set1)
# dir_list1 = [test_set1 + "\\" + files for files in dir_list1]
# dir_list2 = os.listdir(test_set2)
# dir_list2 = [test_set2 + "\\" + files for files in dir_list2]
# dir_list_final = dir_list1 + dir_list2
# print(dir_list_final)
# for filename in dir_list_final:
#     print("file name: " + filename)
#     img = image.load_img(filename, target_size=(223, 221))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0
#     predicted_class = model.predict(img_array)
#     # if isinstance(predicted_class, np.ndarray):
#     #     predicted_class = predicted_class[0][1]
#     # predicted_probabilities.append(predicted_class)
#     # probabilities = model.predict(np.expand_dims(S_resize, axis=0))[0]
#from tensorflow.python import keras

NUM_CLASSES = 2
input_size = (223, 221, 3)


def get_config(input_shape):
    return {
        "filts": [input_shape[-1], [input_shape[-1], 20], [20, 64], [64, 64]],
        "nb_fc_node": 64,
        "gru_node": 64,
        "nb_gru_layer": 2,
        "nb_classes": 1,
    }


class ResidualBlock2D(layers.Layer):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = layers.BatchNormalization()

        self.lrelu = layers.LeakyReLU(alpha=0.3)

        self.conv1 = layers.Conv2D(
            filters=nb_filts[1],
            kernel_size=3,
            padding='same',
            strides=1
        )

        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            filters=nb_filts[1],
            kernel_size=3,
            padding='same',
            strides=1
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = layers.Conv2D(
                filters=nb_filts[1],
                kernel_size=1,
                strides=1
            )

        else:
            self.downsample = False
        self.mp = layers.MaxPooling2D(pool_size=(2, 2))

    def call(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


@register_keras_serializable()
class SpecRNet(Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__()
        config = get_config(input_shape)

        self.first_bn = layers.BatchNormalization()
        self.selu = layers.Activation('selu')
        self.block0 = ResidualBlock2D(nb_filts=config["filts"][1], first=True)
        self.block2 = ResidualBlock2D(nb_filts=config["filts"][2])
        config["filts"][2][0] = config["filts"][2][1]
        self.block4 = ResidualBlock2D(nb_filts=config["filts"][2])
        self.avgpool = layers.GlobalAveragePooling2D()

        self.fc_attention0 = self._make_attention_fc(
            in_features=config["filts"][1][-1], l_out_features=config["filts"][1][-1]
        )
        self.fc_attention2 = self._make_attention_fc(
            in_features=config["filts"][2][-1], l_out_features=config["filts"][2][-1]
        )
        self.fc_attention4 = self._make_attention_fc(
            in_features=config["filts"][2][-1], l_out_features=config["filts"][2][-1]
        )

        self.bn_before_gru = layers.BatchNormalization()
        self.gru = layers.GRU(
            units=config["gru_node"],
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform',
            dropout=0.5,
            recurrent_dropout=0.5
        )

        self.fc1_gru = layers.Dense(
            units=config["nb_fc_node"] * 2
        )

        self.fc2_gru = layers.Dense(
            units=config["nb_classes"],
            activation='sigmoid'
        )
        self.flatten = layers.Flatten()
        self.final_dense = layers.Dense(NUM_CLASSES, activation='sigmoid')

    def _compute_embedding(self, x):
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0)
        y0 = self.fc_attention0(y0)
        y0 = tf.expand_dims(tf.expand_dims(tf.sigmoid(y0), axis=-1), axis=-1)
        reshaped_y0 = tf.keras.backend.reshape(y0, (-1, 1, 1, 20))
        x = x0 * reshaped_y0 + reshaped_y0
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x2 = self.block2(x)
        y2 = self.avgpool(x2)
        y2 = self.fc_attention2(y2)
        y2 = tf.expand_dims(tf.expand_dims(tf.sigmoid(y2), axis=-1), axis=-1)
        reshaped_y2 = tf.keras.backend.reshape(y2, (-1, 1, 1, 64))
        x = x2 * reshaped_y2 + reshaped_y2
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x4 = self.block4(x)
        y4 = self.avgpool(x4)
        y4 = self.fc_attention4(y4)
        y4 = tf.expand_dims(tf.expand_dims(tf.sigmoid(y4), axis=-1), axis=-1)
        reshaped_y4 = tf.keras.backend.reshape(y4, (-1, 1, 1, 64))
        x = x2 * reshaped_y4 + reshaped_y4
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = self.bn_before_gru(x)
        x = self.selu(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = tf.expand_dims(x, axis=1)
        x = self.gru(x)
        x = self.fc1_gru(x[0])
        x = self.fc2_gru(x)
        x = self.flatten(x)
        x = self.final_dense(x)
        return x

    def call(self, x, **kwargs):
        x = self._compute_embedding(x)
        return x

    def _make_attention_fc(self, in_features, l_out_features):
        l_fc = []
        l_fc.append(layers.Dense(units=l_out_features))
        return tf.keras.Sequential(l_fc)


custom_objects = {
    'SpecRNet': SpecRNet,
    'ResidualBlock2D': ResidualBlock2D
}
#     predicted_class = np.argmax(predicted_class)
#     print("Predicted Class:", predicted_class)
#     filename = os.path.splitext(os.path.basename(filename))[0] + ".wav"
#     # print(predicted_class)
#     # print(filename)
#     test_csv = pd.read_csv("D:\\whisper_code\\meta.csv")
#     if filename in test_csv.values:
#         print(filename, "   fill ")
#         # Find the row index where the file name is present in the Excel sheet
#         row_index = test_csv.index[test_csv['file'] == filename[:-4] + ".wav"].tolist()[0]
#         print("row_index", row_index)
#         # Update the 4th column of that row to "run"
#         test_csv.iloc[row_index, test_csv.columns.get_loc('PL_MS_SPECRNET')] = predicted_class
#         print("predicted class: ", predicted_class)
#         # test_csv.iloc[index,3]=predicted_class
#         # index+=1
#         test_csv.to_csv("D:\\whisper_code\\meta.csv", index=False)
#
# # Calculate true positives, false positives, and ROC curve
#
#
# predicted_labels = test_csv['PL_MS_SPECRNET'].tolist()
# true_labels = test_csv['tlabel'].tolist()
# # # fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
# # roc_auc = auc(fpr, tpr)
# # optimal_threshold = thresholds[np.argmax(tpr - fpr)]
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
#
# # Print or visualize the confusion matrix
# print("Confusion Matrix:")
# print(conf_matrix)
#
# test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
#
# print(test_accuracy)

import json
model= joblib.load("output_specRnet.pkl")
dicti={}
def ASVSpoof2019_LA_eval():
    print("LA_evaluation_dataset")
    #test_dir = "D:\\vit btech final year 2023\\Capstone\\Datasets\\release_in_the_wild"
    test_set1 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\bonafide_new"
    test_set2 = "D:\\vit btech final year 2023\\Capstone\\Datasets\\Audiospoof 2019\\LA(1)\\LA\\ASVspoof2019_LA_eval\\spectrograms\\spoof_new"
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
        filename=os.path.splitext(os.path.basename(filename))[0]
        print(filename)
        with open("SPECR_ASV19.txt", "a") as file:
            # Append data to the file
            file.write(f"{filename}: {predicted_class}\n")

ASVSpoof2019_LA_eval()
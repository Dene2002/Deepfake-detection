import os
from keras.src.saving.object_registration import register_keras_serializable
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
#from keras.utils.generic_utils import register_keras_serializable
#from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
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


model = joblib.load("output_specRnet.pkl")

print("testing for ASV spoof 2021")
dicti={}
test_csv = pd.read_csv("D:\\vit btech final year 2023\\Capstone\\Datasets\\ASVSpoof2021\\keys\\CM\\ASV21_SPECR_meta.csv")
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
        img = image.load_img(mel_spec,target_size=(223, 221))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array=img_array/255.0
        predicted_class = model.predict(img_array)
        predicted_class = np.argmax(predicted_class)
        print(filename)
        with open("SPECR_ASV21.txt", "a") as file:
            # Append data to the file
            file.write(f"{filename}: {predicted_class}\n")
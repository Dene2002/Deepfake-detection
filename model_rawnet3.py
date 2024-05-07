
import tensorflow as tf
from keras.layers import Concatenate
from keras.src.saving.object_registration import register_keras_serializable

from tensorflow.keras import layers, Model

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.optimizers.legacy import Adam

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

import numpy as np

from first_iteration import x as l  # Assuming `first_iteration` contains your data loading code

from first_iteration import y as m  # Assuming `first_iteration` contains your data loading code

import warnings



import pickle

warnings.filterwarnings("ignore")

NUM_CLASSES = 2


# Load data using train_test_split

x_train, x_test, y_train, y_test = train_test_split(l, m, stratify=m, test_size=0.3, random_state=0)


# Normalize input data

x_train_norm = np.array(x_train)/255

x_test_norm = np.array(x_test)/255


# Convert labels to one-hot encoded format

y_train_encoded = to_categorical(y_train, NUM_CLASSES)

y_test_encoded = to_categorical(y_test, NUM_CLASSES)


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

    # def call(self, x):
    #     count = 0
    #     residual = self.residual(x)
    #     out = self.conv1(x)
    #     out = self.relu(out)
    #     out = self.bn1(out)
    #
    #     spx = tf.split(out, self.width, axis=3)  # Split along the channel dimension
    #     # sp = spx[0]  # Initialize sp with the first split
    #     # print("first iteration for sp ki value",spx[0])
    #     convs = [layers.Conv2D(filters=self.width, kernel_size=1) for _ in range(self.nums)]
    #
    #     for i in range(self.nums):
    #         if i == 0:
    #             sp = spx[i]
    #             count = count+1
    #             print("sp 0 ki value times",count,"SP: ",sp)
    #         else:
    #             print("output of",i ,"iteration =",spx[i])
    #             spx_i_adjusted = layers.Conv2D(filters=self.width, kernel_size=1)(spx[i])
    #             sp = tf.keras.layers.Add()([sp, spx_i_adjusted])
    #         sp = self.convs[i](sp)
    #         sp = self.relu(sp)
    #         sp = self.bns[i](sp)
    #         if i == 0:
    #             out = sp
    #         else:
    #             out = Concatenate(axis=-1)([out, sp])
    #
    #     out = self.conv3(out)
    #     out = self.relu(out)
    #     out = self.bn3(out)
    #
    #     out += residual
    #
    #     if self.mp:
    #         out = self.mp(out)
    #
    #     return out


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


model = RawNet3(input_shape=(223, 221, 3))  # Adjust input shape accordingly

model.compile(optimizer=Adam(learning_rate=0.0001), loss=binary_crossentropy, metrics=['accuracy'])

sample_input = np.zeros((1,) + (223,221,3), dtype=np.float32)  # Create a sample input with correct shape
output_tensor = model(sample_input)

model.summary()

history = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), epochs=10, batch_size=8)

# Save the model


import joblib
joblib.dump(model,'output_rawnet3.pkl')
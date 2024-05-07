import tensorflow as tf
from keras.layers import Concatenate

from tensorflow.keras import layers, Model

from tensorflow.keras.losses import *

from tensorflow.keras.optimizers import *

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical

from first_iteration import x as l  # Assuming `first_iteration` contains your data loading code

from first_iteration import y as k  # Assuming `first_iteration` contains your data loading code

import warnings

import numpy

warnings.filterwarnings("ignore")

NUM_CLASSES = 2

# Load data using train_test_split

x_train, x_test, y_train, y_test = train_test_split(l, k, stratify=k, test_size=0.3, random_state=0)

# Normalize input data

x_train_norm = numpy.array(x_train) / 255.0

x_test_norm = numpy.array(x_test) / 255.0

# Convert labels to one-hot encoded format

y_train_encoded = to_categorical(y_train, NUM_CLASSES)

y_test_encoded = to_categorical(y_test, NUM_CLASSES)




class PreEmphasis(layers.Layer):

    def __init__(self, coef=0.97):
        super(PreEmphasis, self).__init__()

        self.coef = coef

    def call(self, inputs):
        result = inputs - self.coef * tf.pad(inputs, [[0, 0], [1, 0], [1, 0], [0, 0]], mode='REFLECT')[:, :-1, :-1, :]

        return result


class Bottle2neck(layers.Layer):
    def __init__(self, inplanes, planes, dilation=None, scale=4, pool=False):
        super(Bottle2neck, self).__init__()
        self.inplanes = inplanes
        width = planes // scale
        self.conv1 = layers.Conv2D(width * scale, kernel_size=1)
        self.bn1 = layers.BatchNormalization()
        self.nums = scale - 1
        self.convs = [layers.Conv2D(width, kernel_size=1, dilation_rate=dilation, padding='same') for _ in
                      range(self.nums)]
        self.bns = [layers.BatchNormalization() for _ in range(self.nums)]
        self.conv3 = layers.Conv2D(planes, kernel_size=1)
        self.bn3 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.width = width
        self.mp = layers.MaxPool2D(pool) if pool else None
        self.afms = AlphaFeatureMapScaling(planes)
        if inplanes != planes:
            self.residual = layers.Conv2D(planes, kernel_size=1, strides=1, use_bias=False)
        else:
            self.residual = layers.Lambda(lambda x: x)
        self.conv_adjust = layers.Conv2D(filters=self.width, kernel_size=1)

        #self.convs = [layers.Conv2D(filters=self.width, kernel_size=1) for _ in range(self.nums)]

    def call(self, x):
        count = 0
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = tf.split(out, self.width, axis=3)  # Split along the channel dimension
        sp = spx[0]  # Initialize sp with the first split
        #self.convs = [layers.Conv2D(filters=self.width, kernel_size=1) for _ in range(self.nums)]

        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                spx[i] = self.conv_adjust(spx[i])  # Adjust the number of channels using 1x1 convolution
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = Concatenate(axis=-1)([out, sp])

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out += residual

        if self.mp:
            out = self.mp(out)

        return out


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

        self.layer1 = Bottle2neck(1024 // 4, 1024, dilation=2, scale=model_scale, pool=(5, 5))

        self.layer2 = Bottle2neck(1024, 1024, dilation=3, scale=model_scale, pool=(3, 3))

        self.layer3 = Bottle2neck(1024, 1024, dilation=4, scale=model_scale)

        self.layer4 = layers.Conv2D(3 * 1024, kernel_size=1)
        self.attention = tf.keras.Sequential([
            layers.Conv2D(filters=128, kernel_size=1),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Conv2D(filters=1536, kernel_size=1),
            layers.BatchNormalization(),
            layers.Softmax(axis=-1)
        ])
        self.conv_w = layers.Conv2D(filters=3072, kernel_size=1)

        self.bn5 = layers.BatchNormalization()

        self.fc6 = layers.Dense(256)

        self.fc2 = layers.Dense(NUM_CLASSES, activation="softmax")

        self.flatten = layers.Flatten()

    def call(self, inputs):
        x = self.preprocess(inputs)
        x = tf.abs(self.conv1(x))

        if self.log_sinc:
            x = tf.math.log(tf.abs(x) + 1e-8)

        if self.norm_sinc == "mean":

            x = x - tf.math.reduce_mean(x, axis=-1, keepdims=True)

        elif self.norm_sinc == "cmn":

            x = x - tf.math.reduce_mean(x, axis=[1, 2], keepdims=True)

        if self.summed:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(layers.MaxPool2D((3, 3))(x1) + x2)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
        x = self.layer4(tf.concat([layers.MaxPool2D((3, 3))(x1), x2, x3], axis=-1))
        x = self.relu(x)
        # t = x.shape[-1]
        if self.context:
            global_x = tf.concat(
                [x, tf.tile(tf.reduce_mean(x, axis=[1, 2], keepdims=True), [1, x.shape[1], x.shape[2], 1]),
                 tf.tile(tf.math.sqrt(tf.math.reduce_variance(x, axis=[1, 2], keepdims=True)),
                         [1, x.shape[1], x.shape[2], 1])], axis=-1)
        else:
            global_x = x
        w = self.attention(global_x)
        w = self.conv_w(w)
        mu = tf.reduce_sum(x * w ,axis=[1, 2])
        sg = tf.math.sqrt(tf.reduce_sum((x ** 2) * w, axis=[1, 2]) - mu ** 2)
        x = tf.concat([mu, sg], axis=-1)
        x = self.bn5(x)
        x = self.flatten(x)
        x = self.fc6(x)
        if self.out_bn:
            x = self.bn5(x)
        x = self.fc2(x)
        print(x.shape)
        return x


model = RawNet3(input_shape=(223, 221, 3))  # Adjust input shape accordingly

model.compile(optimizer=Adam(learning_rate=0.0001,decay=0.00005), loss=binary_crossentropy, metrics=['accuracy'])

sample_input = numpy.zeros((1,) + (223, 221, 3), dtype=numpy.float32)  # Create a sample input with correct shape
output_tensor = model(sample_input)

model.summary()

history = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), epochs=10,
                    batch_size=8)
#import joblib
model.save_weights("output_rawnet3_weights.h5")





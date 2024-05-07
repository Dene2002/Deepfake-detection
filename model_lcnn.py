from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Flatten, Reshape
from keras.layers import Bidirectional, LSTM
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
from first_iteration import x, y
import matplotlib.pyplot as plt
from keras.regularizers import l2
import tensorflow as tf

NUM_CLASSES = 2

# Load data and preprocess
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)
x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255
y_train_encoded = to_categorical(y_train, NUM_CLASSES)
y_test_encoded = to_categorical(y_test, NUM_CLASSES)

from keras.layers import Layer, MaxPooling2D


model = Sequential()


# Define the model
model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=(223, 221, 3),
                 kernel_regularizer=l2(0.0001)))
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
model.add(MaxFeatureMap2D())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Conv2D(96, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(96, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64, (1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.0001)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.7))
model.add(Flatten())
model.add(Reshape((6, -1)))
model.add(Bidirectional(LSTM(units=(80//16) * 32, return_sequences=True)))
model.add(Bidirectional(LSTM(units=(80//16) * 32, return_sequences=True)))
model.add(Flatten())
model.add(Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=l2(0.0001)))

# Compile the model
optimizer = Adam(learning_rate=0.0001, decay=0.0001)
model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
model.summary()

# Train the model
output_lcnn = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), batch_size=8,
                   epochs=10)

#Graph
acc = output_lcnn.history['accuracy']
val_acc = output_lcnn.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy for LCNN model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()
from keras.utils import plot_model
#saving the model
model.save('output_lcnn.h5')
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

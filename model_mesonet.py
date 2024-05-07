from first_iteration import x,y
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

NUM_CLASSES=2

# Load data and preprocess
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=0)
x_train_norm = np.array(x_train) / 255
x_test_norm = np.array(x_test) / 255
y_train_encoded = to_categorical(y_train, NUM_CLASSES)
y_test_encoded = to_categorical(y_test, NUM_CLASSES)

from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate, ReLU, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

NUM_CLASSES=2


def InceptionLayer1(input_tensor):
    # Branch 1
    branch1 = Conv2D(1, (1, 1), padding='same', use_bias=False)(input_tensor)

    # Branch 2
    branch2 = Conv2D(4, (1, 1), padding='same', use_bias=False)(input_tensor)
    branch2 = Conv2D(4, (3, 3), padding='same', use_bias=False)(branch2)

    # Branch 3
    branch3 = Conv2D(4, (1, 1), padding='same', use_bias=False)(input_tensor)
    branch3 = Conv2D(4, (3, 3), padding='same', dilation_rate=2, use_bias=False)(branch3)

    # Branch 4
    branch4 = Conv2D(2, (1, 1), padding='same', use_bias=False)(input_tensor)
    branch4 = Conv2D(2, (3, 3), padding='same', dilation_rate=3, use_bias=False)(branch4)

    # Concatenate branches
    output = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    output = BatchNormalization()(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)

    return output

def InceptionLayer2(input_tensor):
    # Branch 1
    branch1 = Conv2D(2, (1, 1), padding='same', use_bias=False)(input_tensor)

    # Branch 2
    branch2 = Conv2D(4, (1, 1), padding='same', use_bias=False)(input_tensor)
    branch2 = Conv2D(4, (3, 3), padding='same', use_bias=False)(branch2)

    # Branch 3
    branch3 = Conv2D(4, (1, 1), padding='same', use_bias=False)(input_tensor)
    branch3 = Conv2D(4, (3, 3), padding='same', dilation_rate=2, use_bias=False)(branch3)

    # Branch 4
    branch4 = Conv2D(2, (1, 1), padding='same', use_bias=False)(input_tensor)
    branch4 = Conv2D(2, (3, 3), padding='same', dilation_rate=3, use_bias=False)(branch4)

    # Concatenate branches
    output = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    output = BatchNormalization()(output)
    output = MaxPooling2D(pool_size=(2, 2))(output)

    return output


# Define input layer
input_layer = Input(shape=(223, 221, 3))

# Build Inception layers
inception1_output = InceptionLayer1(input_layer)
inception2_output = InceptionLayer2(inception1_output)

# Normal Layer
conv3 = Conv2D(16, (5, 5), padding='same', use_bias=False)(inception2_output)
relu3 = ReLU()(conv3)
bn3 = BatchNormalization()(relu3)
pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

conv4 = Conv2D(16, (5, 5), padding='same', use_bias=False)(pool3)
relu4 = ReLU()(conv4)
bn4 = BatchNormalization()(relu4)
pool4 = MaxPooling2D(pool_size=(4, 4))(bn4)

flatten = Flatten()(pool4)
dense1 = Dense(1024, activation='relu',kernel_initializer='random_normal')(flatten)
dropout = Dropout(0.5)(dense1)
output_layer = Dense(NUM_CLASSES, activation='softmax')(dropout)

# Define model
model = Model(inputs=input_layer, outputs=output_layer)

optimizer = Adam(learning_rate=0.0001, decay=0.0001)

# Compile model
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

output_mesonet = model.fit(x_train_norm, y_train_encoded, validation_data=(x_test_norm, y_test_encoded), batch_size=8,
                   epochs=10)

#Graph
acc = output_mesonet.history['accuracy']
val_acc = output_mesonet.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, '-', label='Training Accuracy')
plt.plot(epochs, val_acc, ':', label='Validation Accuracy')
plt.title('Training and Validation Accuracy for LCNN model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.plot()
plt.show()

#saving the model
model.save('output_mesonet.h5')


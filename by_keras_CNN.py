from __future__ import absolute_import, division, print_function, unicode_literals
import abc, absl, functools, inspect, copy, json
import pathlib, sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

assert tf.executing_eagerly()


########################################################################################################################
# loading DATASET
#
_path = pathlib.Path.cwd().joinpath('data', 'mnist.npz').as_posix()
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data(path=_path)



########################################################################################################################
# PREPROCESSING
#
# reshaping
assert(X_train.shape[1] == X_test.shape[1] and
       X_train.shape[2] == X_test.shape[2])
# assuming 'channels_last':
input_shape = (X_train.shape[1], X_train.shape[2], 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
assert(X_train.shape == (60000, 28, 28, 1) and
       X_test.shape == (10000, 28, 28, 1))
# normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
#
#
# transforming labels:int  into one hot encoding
num_category = 10
Y_train = keras.utils.to_categorical(y_train, num_category)
Y_test = keras.utils.to_categorical(y_test, num_category)
#




#
j = 2
########################################################################################################################
# Model
#
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=input_shape, dtype=tf.float32))
# feature extraction ---------------------------------------------------------------------------------------------------
if j >= 0:
    model.add(keras.layers.Conv2D(filters=24, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D())
if j >= 1:
    model.add(keras.layers.Conv2D(filters=48, kernel_size=(5, 5), strides=(1, 1),  padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D())
if j >= 2:
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1),  padding='same', activation='relu'))
    model.add(keras.layers.MaxPool2D(padding='same'))
# classification -------------------------------------------------------------------------------------------------------
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
#
#
model.compile(optimizer=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004),
              loss="categorical_crossentropy",
              metrics=["accuracy"])




########################################################################################################################
# TRAINING
#
history = model.fit(x=X_train, y=Y_train, epochs=5)








########################################################################################################################
# EVALUATE
#
test_scores = model.evaluate(x=X_test, y=Y_test, verbose=0)
#
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])


########################################################################################################################
# predict
#
pred = np.argmax(model.predict(x=X_test), axis=1)
idx_where_fault = np.argwhere(y_test != pred)


# visual test: plotting all wrongs
import matplotlib.pyplot as plt
while len(idx_where_fault) > 0:

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        if len(idx_where_fault) > 0:
            idx = idx_where_fault[0]
            idx_where_fault = np.delete(idx_where_fault, 0, 0)
            imag = X_test[idx] * 255
            imag.astype(int)
            imag = imag.squeeze()
            plt.imshow(imag, cmap='gray', interpolation='none')
            plt.title("#{}: true={}, pred={}".format(idx, y_test[idx], pred[idx]), fontsize='xx-small')
            plt.xticks([])
            plt.yticks([])
        else:
            break
    plt.show()

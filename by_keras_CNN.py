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
# loading dataset
#
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data() # stores it at '~/.keras/datasets/'
#
# visual test
import matplotlib.pyplot as plt
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
plt.show()


########################################################################################################################
# transforming labels into one hot encoding
#
# set number of categories
num_category = 10
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)





########################################################################################################################
#reshaping
#this assumes our data format
#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while
#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
#
assert(X_train.shape[1] == X_test.shape[1] and X_train.shape[2] == X_test.shape[2])
#
# assuming 'channels_first':
input_shape = (1, X_train.shape[1], X_train.shape[2])
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
#
# normalize
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape) # X_train shape: (60000, 1, 28, 28)
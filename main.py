from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib, sys, os
import abc, functools, inspect
import matplotlib.pyplot as plt
import numpy as np
import absl
import tensorflow as tf
import tensorflow_datasets as tfds

assert (tf.executing_eagerly() is True), 'ERROR: tf is NOT executing eagerly'

#####################################################################
#####################################################################
# importing dataset using KERAS #####################################

ks = tf.keras.datasets.mnist.load_data()
# ks_train, ks_test = ks
# (x_train, y_train), (x_test, y_test) = ks_train, ks_test
# x_train, x_test = x_train / 255.0, x_test / 255.0


####################################################################
####################################################################
# importing dataset using TFDS #####################################

mnist_builder = tfds.builder("mnist")
mnist_builder.download_and_prepare()
mnist_train = mnist_builder.as_dataset(split=tfds.Split.TRAIN)
assert isinstance(mnist_train, tf.data.Dataset)







# cwd = pathlib.Path.cwd()
# p = cwd + '\\..\\,,,' + '\\tensorflow_datasets' + '\\mnist\\1.0.0\\'
# filedir = 'C:\\Users\\roble\\tensorflow_datasets\\mnist\\1.0.0\\'
# filenames = []
# dataset = tf.data.TFRecordDataset(filenames)




#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)
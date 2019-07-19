from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import copy

name, split = 'mnist', tfds.Split.TRAIN

# dataset
data = list()
data.append(tfds.builder(name=name))
data.append(data[-1])
data[-1].download_and_prepare()
data.append(data[-1].as_dataset(split=split))
assert isinstance(data[-1], tf.data.Dataset)

# model
model = list()
model.append(keras.Input(shape=(28, 28)))
model.append(keras.layers.Dense(64, activation=tf.nn.relu)(model[-1]))
model.append(keras.layers.Dense(64, activation=tf.nn.relu)(model[-1]))
model.append(keras.layers.Dense(10, activation=tf.nn.softmax)(model[-1]))
model.append(keras.Model(inputs=model[0], outputs=model[-1]))
assert isinstance(model[-1], keras.Model)

model.append(model[-1])
model[-1].compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
assert isinstance(model[-1], keras.Model)



# model.append(model[-1])
model[-1].fit(x=data[-1], epochs=5)




#model.evaluate(mnist)


# isinstance(tf.data.Dataset, tf.python.)
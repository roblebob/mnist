from __future__ import absolute_import, division, print_function, unicode_literals
import abc, absl, functools, inspect, copy, json
import pathlib, sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

assert tf.executing_eagerly()


class Mnist(tf.keras.Model):

    def __init__(self, *args, **kwargs):
        """
        constructer:
            host the process history of the model creating within succeding list entries (time steps)
                    o[0]   :  first element
                    o[-1]  :  last element  (latest)
            hosts the data-core  (... is going to)
                    x[...]
        """
        super().__init__(*args, **kwargs)
        self.o = [tf.keras.Input(shape=(28, 28, 1), dtype=tf.float32)]
        self.x = {'train': None, 'test': None, 'valid': None}


    def model(self):
        """defining process order"""
        self.o.append(tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu)(self.o[-1]))
        self.o.append(tf.keras.layers.MaxPooling2D((2, 2))(self.o[-1]))
        self.o.append(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)(self.o[-1]))
        self.o.append(tf.keras.layers.MaxPooling2D((2, 2))(self.o[-1]))
        self.o.append(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu)(self.o[-1]))
        self.o.append(tf.keras.layers.Flatten()(self.o[-1]))
        self.o.append(tf.keras.layers.Dense(64, activation=tf.nn.relu)(self.o[-1]))
        self.o.append(tf.keras.layers.Dense(10, activation=tf.nn.softmax)(self.o[-1]))
        self.o.append(tf.keras.Model(inputs=self.o[0], outputs=self.o[-1]))
        assert isinstance(self.o[-1], tf.keras.Model)


    def display(self):
        """displays the most recent model specification"""
        assert(isinstance(self.o[-1], tf.keras.Model))
        self.o[-1].summary()
        tf.keras.utils.plot_model(self.o[-1], 'data/model.png', show_shapes=True)


    def data_core(self):
        """"""
        with pathlib.Path.cwd().joinpath('data', 'sources', '1.0.0', 'dataset_info.json').as_posix() as json_file:
            json_str = json_file.read()
        fresh_model = tf.keras.models.model_from_json(json_str)


    def data_0(self):
        """
        returns tuple of nparrays:
            ....
            cached outside projectfolder, namely:   ~/.keras
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        train = tf.data.Dataset.from_tensor_slices(
            (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32),
             tf.cast(y_train, tf.int64)))
        test = tf.data.Dataset.from_tensor_slices(
            (tf.cast(x_test[..., tf.newaxis] / 255, tf.float32),
             tf.cast(y_test, tf.int64)))
        assert(isinstance(train, tf.data.Dataset) and isinstance(test, tf.data.Dataset))
        return {'train' : train, 'test' : test}


    def data_1(self, split=tfds.Split.ALL):
        builder = tfds.builder(name="mnist", data_dir=pathlib.Path.cwd().joinpath('data'))
        builder.download_and_prepare()
        self.x = builder.as_dataset(split=split)


    def data_TFRecord(self):
        path = pathlib.Path.cwd().joinpath('data','sources','1.0.0')
        filenames = [x for x in path.glob('**/*') if x.is_file()]
        filenames = [x.as_posix() for x in iter(filenames)]

        filenames_tfrecords = [x for x in iter(filenames) if 'train.tfrecord' in x]
        dataset = tf.data.TFRecordDataset(filenames_tfrecords)

        with open(path.joinpath('dataset_info.json').as_posix(), 'r') as myfile:
            s = myfile.read()

        ss = json.loads(s)
        sss = tf.io.decode_json_example(s)


    def training(self, data: tf.data.Dataset):
        """ """
        self.o.append(self.o[-1].compile(optimizer='adam',
                                         loss='sparse_categorical_crossentropy',
                                         metrics=['accuracy']))
        self.o[-1].fit(data, labels, batch_size=32, epochs=5)
        assert(isinstance(dataset, tf.data.Dataset))


    def training_0(history: list, dataset: dict, shuffle=1000, batch=32):

        model = history[-1].compile(optimizer='adam',
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])
        data = dataset['train'].shuffle(shuffle).batch(batch)
        assert(isinstance(model, keras.Model) and isinstance(data, tf.data.Dataset))


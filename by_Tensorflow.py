from __future__ import absolute_import, division, print_function, unicode_literals
import abc, absl, functools, inspect, copy, json
import pathlib, sys, os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

assert tf.executing_eagerly()


# setup dataset --------------------------------------------------------------------------------------------------
def one_hot_encoded(idx: int):
    assert(1 <= idx <= 10)
    return [(1.0 * (i == idx)) for i in (np.arange(10) + 1)]


def display(_):
    """displays the model specified"""
    assert(isinstance(_, tf.keras.Model))
    _.summary()
    tf.keras.utils.plot_model(_, 'data/model.png', show_shapes=True)


# setup dataset --------------------------------------------------------------------------------------------------
def setup_dataset_input_pipeline(split='train') -> tf.data.Dataset:
    """"""
    _ = tfds.builder(name="mnist", data_dir=pathlib.Path.cwd().joinpath('data'))
    # TODO: rewrite def of "data_dir" using "tf.io" instead of "pathlib"

    _.download_and_prepare()
    return _.as_dataset(split=split)


_ds = {'train': setup_dataset_input_pipeline(split='train'),
       'test': setup_dataset_input_pipeline(split='test')}


#_ds['train'] = _ds['train'].shuffle(1024).batch(1)

# prefetch will enable the input pipeline to asynchronously fetch batches while your model is training.
_ds['train'] = _ds['train'].prefetch(tf.data.experimental.AUTOTUNE)







########################################################################################################################
# preparing model

def setup_model(input_shape: tf.TensorShape) -> tf.keras.Model:
    """setup_model_process_block_precompiled"""
    # setup input
    _input = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    # setup process block
    _ = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.relu)(_input)
    _ = tf.keras.layers.MaxPooling2D((2, 2))(_)
    _ = tf.keras.layers.Conv2D(filters=56, kernel_size=(3, 3), activation=tf.nn.relu)(_)
    _ = tf.keras.layers.MaxPooling2D((2, 2))(_)
    _ = tf.keras.layers.Conv2D(filters=56, kernel_size=(3, 3), activation=tf.nn.relu)(_)
    _ = tf.keras.layers.Flatten()(_)
    _ = tf.keras.layers.Dense(filters=56, activation=tf.nn.relu)(_)
    _ = tf.keras.layers.Dense(filters=10, activation=tf.nn.softmax)(_)
    return tf.keras.Model(inputs=_input, outputs=_)


_model = setup_model(input_shape=_ds['train'].output_shapes['image'])


########################################################################################################################
# Specify the training configuration (optimizer, loss, metrics)

_model.compile(loss='sparse_categorical_crossentropy',
               optimizer=keras.optimizers.RMSprop(),
               metrics=['accuracy'])


history = _model.fit(_ds['train'], epochs=5)

test_scores = _model.evaluate(_ds['train'], verbose=0)

print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
















    # def data_TFRecord(self):
    #     path = pathlib.Path.cwd().joinpath('data','mnist','1.0.0')
    #     filenames = [x for x in path.glob('**/*') if x.is_file()]
    #     filenames = [x.as_posix() for x in iter(filenames)]
    #
    #     filenames_tfrecords = [x for x in iter(filenames) if 'train.tfrecord' in x]
    #     dataset = tf.data.TFRecordDataset(filenames_tfrecords)
    #
    #     with open(path.joinpath('dataset_info.json').as_posix(), 'r') as myfile:
    #         s = myfile.read()
    #
    #     ss = json.loads(s)
    #     sss = tf.io.decode_json_example(s)




# def main():
#     mnist = Mnist()
#     mnist.display()
#
#
# if __name__ == '__main__':
#     main()


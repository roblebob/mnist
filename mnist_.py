from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

assert(tf.executing_eagerly())


#dataset = sources.data.data_0()
dataset = sources.data.data_TFRecord()
# Fetch and format the sources data .........................................................
# (mnist_images, mnist_labels), _ = keras.datasets.sources.load_data()
#
# dataset = tf.data.Dataset.from_tensor_slices(
#     (tf.cast(mnist_images[..., tf.newaxis]/255, tf.float32),
#      tf.cast(mnist_labels, tf.int64)))




#m = sources.model.definition_0()
#sources.model.display(m)
#h = sources.model.compilation(h)



# dataset = dataset.shuffle(1000).batch(32)
#
#
# # Build the model ........................................................................

#
#
#
#
# for images,labels in dataset.take(1):
#   print("Logits: ", mnist_model(images[0:1]).numpy())
#
#
# optimizer = tf.keras.optimizers.Adam()
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss_history = []
#
#
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         logits = mnist_model(images, training=True)
#
#         # Add asserts to check the shape of the output.
#         tf.debugging.assert_equal(logits.shape, (32, 10))
#
#         loss_value = loss_object(labels, logits)
#
#     loss_history.append(loss_value.numpy().mean())
#     grads = tape.gradient(loss_value, mnist_model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
#
#
# def train():
#   for epoch in range(3):
#     for (batch, (images, labels)) in enumerate(dataset):
#       train_step(images, labels)
#     print ('Epoch {} finished'.format(epoch))
#
#
# train()
#
# import matplotlib.pyplot as plt
#
# plt.plot(loss_history)
# plt.xlabel('Batch #')
# plt.ylabel('Loss [entropy]')
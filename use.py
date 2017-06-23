import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

from model import DCGAN

model = DCGAN()
saver = tf.train.Saver()
mnist = input_data.read_data_sets("MNIST_data/")

with tf.Session() as sess:
    saver.restore(sess, "model/mnist-demo.ckpt-500")
    print("Model restored.")

    random_array = np.random.rand(model.batch_size, 100)
    mask = np.full(shape=(model.batch_size,), fill_value=0)
    classes = np.linspace(0, model.batch_size - 1, model.batch_size).astype(np.int32)
    classes[:model.n_classes] = list(range(model.n_classes))

    feed = {
        model.label: classes,
        model.mask: mask,
        model.random: random_array
    }

    output = sess.run(
        [model.generations],
        feed_dict=feed)

images = output[0]
image = images[0].flatten()

train = mnist.train.images[mnist.train.labels == 0]

bestnorm = float("inf")
best_i = -1
for i in range(train.shape[0]):
    norm = np.linalg.norm(image - train[i])
    if norm < bestnorm:
        bestnorm = norm
        best_i = i

print(bestnorm)
a = np.reshape(image, [28, 28])
b = np.reshape(train[best_i], [28, 28])
plt.imshow(1 - np.hstack([a, b]), cmap="Greys")
plt.show()

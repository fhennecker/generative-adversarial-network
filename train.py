import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

from model import DCGAN

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/")

model = DCGAN()
sess.run(tf.global_variables_initializer())

for i in range(int(1e4)):
    real, classes = mnist.train.next_batch(10)
    real = np.reshape(real, [10, 28, 28, 1])
    mask = np.random.randint(0, 10, (10,))
    random = np.random.rand(10, 100)

    feed = {
        model.label: classes,
        model.real_images: real,
        model.mask: mask,
        model.random: random
    }

    sess.run(model.discriminator_train_step, feed_dict=feed)
    sess.run(model.generator_train_step, feed_dict=feed)

    if i % 100 == 0:
        saver = tf.train.Saver()
        saver.save(sess, os.path.join("log", "model.ckpt"), i)

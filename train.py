import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

from model import DCGAN

try:
    os.mkdir("summaries")
except FileExistsError:
    pass
try:
    os.mkdir("model")
except FileExistsError:
    pass

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/")

model_name = "first"
model = DCGAN()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter('summaries/' + model_name)

d_loss_summary = tf.summary.scalar('Losses/discriminator', model.discriminator_loss)
g_loss_summary = tf.summary.scalar('Losses/generator', model.generator_loss)
image_input_summary = tf.summary.image('Input', model.real_images)
gen_image_summary = tf.summary.image('Generated', model.generations)
summaries = tf.summary.merge_all()

for i in range(int(1e4)):
    real, classes = mnist.train.next_batch(10)
    real = np.reshape(real, [10, 28, 28, 1])
    mask = np.random.randint(0, 1, (10,))
    random = np.random.rand(10, 100)

    feed = {
        model.label: classes,
        model.real_images: real,
        model.mask: mask,
        model.random: random
    }

    summary, _, _ = sess.run([
        summaries, model.discriminator_train_step, model.generator_train_step],
        feed_dict=feed)
    summary_writer.add_summary(summary, i)

    if i % 100 == 0:
        saver.save(sess, os.path.join("model/", model_name + ".ckpt"), i)
        print('Saved model.')

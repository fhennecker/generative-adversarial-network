import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import sys
import random

from model import DCGAN

try:
    os.mkdir("summaries")
except FileExistsError:
    pass
try:
    os.mkdir("model")
except FileExistsError:
    pass

if len(sys.argv) < 2:
    print("Provide model name ad first argument (e.g. %s my_model)" % sys.argv[0])
    sys.exit(-1)
else:
    model_name = sys.argv[1]


sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/")

model = DCGAN()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter('summaries/' + model_name, sess.graph)

d_loss_summary = tf.summary.scalar('Losses/discriminator', model.discriminator_loss)
g_loss_summary = tf.summary.scalar('Losses/generator', model.generator_loss)
image_input_summary = tf.summary.image('Input', model.real_images)
gen_image_summary = tf.summary.image('Generated', model.generations)
summaries = tf.summary.merge_all()

for i in range(int(1e6)):
    real, classes = mnist.train.next_batch(model.batch_size)
    real = np.reshape(real, [model.batch_size, 28, 28, 1])

    # mask = np.random.randint(0, 2, (model.batch_size,))
    mask = np.full(shape=(model.batch_size,), fill_value=random.randint(0, 1))

    random_array = np.random.rand(model.batch_size, 100)

    feed = {
        model.label: classes,
        model.real_images: real,
        model.mask: mask,
        model.random: random_array
    }

    summary, _, = sess.run(
        # [summaries, model.discriminator_train_step, model.generator_train_step],
        [summaries, model.discriminator_train_step],
        feed_dict=feed)

    if i % 50:
        summary_writer.add_summary(summary, i)
    if i % 5000 == 0:
        saver.save(sess, os.path.join("model/", model_name + ".ckpt"), i)
        print('Saved model.')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import sys
import random

from model import DCGAN

ONLY_LABEL = None
FULL_MASK = False

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

model = DCGAN()
sess.run(tf.global_variables_initializer())

mnist = input_data.read_data_sets("MNIST_data/")

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter('summaries/' + model_name, sess.graph)

d_loss_summary = tf.summary.scalar('Losses/discriminator', model.discriminator_loss)
g_loss_summary = tf.summary.scalar('Losses/generator', model.generator_loss)
gen_image_summary = tf.summary.image('Generated', model.generations, max_outputs=10)
summaries = tf.summary.merge_all()

X = mnist.train.images[mnist.train.labels == (ONLY_LABEL or 1)]
Y = np.full(shape=(X.shape[0],), fill_value=ONLY_LABEL)

for i in range(int(1e6)):

    if ONLY_LABEL:
        # Select from ONLY_LABEL
        real = X[np.random.randint(X.shape[0], size=model.batch_size), :]
        classes = np.full(shape=(model.batch_size,), fill_value=ONLY_LABEL)
    else:
        # Select from full MNIST
        real, classes = mnist.train.next_batch(model.batch_size)

    # Resize real to a 2D array (was a 1D vector)
    real = np.reshape(real, [model.batch_size, 28, 28, 1])

    if FULL_MASK:
        mask = np.full(shape=(model.batch_size,), fill_value=random.randint(0, 1))
    else:
        mask = np.random.randint(0, 2, (model.batch_size,))

    random_array = np.random.rand(model.batch_size, 100)

    if i % 50:
        # when writing to tensorboard generate all the digits,
        # ordered in the 10 first items of the batch
        mask[:10] = 0
        classes[:10] = list(range(10))

    min_lr = 0.00001
    max_lr = 0.1
    anneal_steps = 1e5
    learning_rate = max(min_lr, max_lr-(max_lr-min_lr)*(i/anneal_steps))
    
    feed = {
        model.label: classes,
        model.real_images: real,
        model.mask: mask,
        model.random: random_array,
        model.gen_learning_rate: learning_rate,
        model.disc_learning_rate: learning_rate,
    }

    summary, _, _ = sess.run(
        [summaries, model.discriminator_train_step, model.generator_train_step],
        feed_dict=feed)

    if i % 50:
        summary_writer.add_summary(summary, i)
    if i % 5000 == 0:
        saver.save(sess, os.path.join("model/", model_name + ".ckpt"), i)
        print('Saved model.')

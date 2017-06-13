import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import batch_norm

LEARNING_RATE = 0.0002


class DCGAN():
    def __init__(self, batch_size=128, n_classes=10, image_size=28):
        with tf.variable_scope("dcgan"):
            self.batch_size = batch_size
            self.n_classes = n_classes
            self.image_size = image_size

            self.label = tf.placeholder(tf.int32, [self.batch_size], name='label')
            self.label_onehot = tf.one_hot(self.label, self.n_classes)
            self.label_map = tf.reshape(
                self.label_onehot, [self.batch_size, 1, 1, self.n_classes])

            self.real_images = tf.placeholder(
                tf.float32,
                [self.batch_size, self.image_size, self.image_size, 1],
                name='real_images')

            self.mask = tf.placeholder(tf.float32, [self.batch_size], 'mask')

            with tf.variable_scope("generate"):
                self._init_generate()
            with tf.variable_scope("discriminate"):
                self._init_discriminate()
            with tf.variable_scope("losses"):
                self._init_losses()

    def _init_generate(self):
        self.random = tf.placeholder(tf.float32, [self.batch_size, 100])

        # TODO batch normalise
        h1 = batch_norm(slim.fully_connected(
            tf.concat([self.random, self.label_onehot], 1),
            512))

        h2 = batch_norm(slim.fully_connected(
            tf.concat([h1, self.label_onehot], 1),
            128 * 7 * 7))
        h2 = tf.reshape(h2, [self.batch_size, 7, 7, 128])
        h2 = tf.concat(
            [h2, self.label_map * tf.ones([self.batch_size, 7, 7, self.n_classes])],
            3)

        c1 = slim.conv2d_transpose(h2, 64, [5, 5], 2, normalizer_fn=slim.batch_norm)
        c1 = tf.concat(
            [c1, self.label_map * tf.ones([self.batch_size, 14, 14, self.n_classes])],
            3)

        # No batchnorm here on purpose
        self.generations = tf.nn.sigmoid(
            slim.conv2d_transpose(c1, 1, [5, 5], 2))

    def _init_discriminate(self):
        im_mask = tf.tile(
            tf.reshape(self.mask, [self.batch_size, 1, 1, 1]),
            [1, 28, 28, 1])

        # No batchnorm here on purpose
        input_images = self.real_images * im_mask + self.generations * (1 - im_mask)
        # Convolution 1
        conv1 = slim.conv2d(
            input_images,
            num_outputs=32, kernel_size=[5, 5],
            stride=[2, 2], padding='Valid',
            normalizer_fn=slim.batch_norm,
        )
        conv1_shape = conv1.get_shape()[1:3]
        level1_label_map = self.label_map * tf.ones(
            [self.batch_size, conv1_shape[0], conv1_shape[1], self.n_classes]
        )

        level1 = tf.concat([conv1, level1_label_map], 3)

        # Convolution 2
        conv2 = slim.conv2d(
            level1,
            num_outputs=32, kernel_size=[5, 5],
            stride=[2, 2], padding='Valid',
            normalizer_fn=slim.batch_norm,
        )
        conv2_shape = conv2.get_shape()[1:3]
        level2_label_map = self.label_map * tf.ones(
            [self.batch_size, conv2_shape[0], conv2_shape[1], self.n_classes]
        )

        level2 = tf.concat([conv2, level2_label_map], 3)

        # Level 3
        fc3 = batch_norm(slim.fully_connected(slim.flatten(level2), 200))
        level3 = tf.concat([fc3, self.label_onehot], 1)

        # Level 4
        self.discriminate_output = batch_norm(slim.fully_connected(level3, 1))

        #  self.discriminate_output = tf.nn.softmax(level4)

    def _init_losses(self):
        with tf.variable_scope("generator"):
            # generator loss
            self.generator_loss = tf.reduce_mean(
                (1 - self.mask) *
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.discriminate_output,
                    labels=tf.ones_like(self.discriminate_output)),
                name="loss"
            )
            generator_variables = list(filter(
                lambda v: v.name.startswith('dcgan/generate'),
                tf.trainable_variables())
            )
            self.generator_train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
                self.generator_loss, var_list=generator_variables,
                beta1=0.5, name="train_step",
            )

        with tf.variable_scope("discriminator"):
            self.discriminator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.squeeze(self.discriminate_output),
                    labels=self.mask
                ),
                name="loss"
            )
            discriminator_variables = list(filter(
                lambda v: v.name.startswith('dcgan/discriminate'),
                tf.trainable_variables()))

            self.discriminator_train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
                self.discriminator_loss, var_list=discriminator_variables,
                beta1=0.5, name="train_step",
            )


if __name__ == '__main__':
    DCGAN()

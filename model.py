import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import batch_norm

LEARNING_RATE = 0.0002


class DCGAN():
    def __init__(self, batch_size=128, n_classes=10, image_size=28, image_depth=1):
        assert batch_size >= image_size, "Batch size must be higher than n_classes due to the summary"

        with tf.variable_scope("dcgan"):
            self.batch_size = batch_size
            self.n_classes = n_classes
            self.image_size = image_size
            self.image_depth = image_depth
            self.conv_size = int(self.image_size / 4)

            self.label = tf.placeholder(tf.int32, [self.batch_size], name='label')
            self.label_onehot = tf.one_hot(self.label, self.n_classes)
            self.label_map = tf.reshape(
                self.label_onehot, [self.batch_size, 1, 1, self.n_classes])

            self.real_images = tf.placeholder(
                tf.float32,
                [self.batch_size, self.image_size, self.image_size, self.image_depth],
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

        # input_layer = self.random
        input_layer = tf.concat([self.random, self.label_onehot], axis=1)

        h1 = slim.fully_connected(input_layer, 512)
        h1 = slim.dropout(h1, 0.5)
        h1 = batch_norm(h1)

        h2 = batch_norm(slim.fully_connected(h1, 128 * self.conv_size * self.conv_size))
        h2 = tf.reshape(h2, [self.batch_size, self.conv_size, self.conv_size, 128])

        c1 = slim.conv2d_transpose(
            h2, 64, [5, 5], 2, normalizer_fn=slim.batch_norm,
            padding="SAME",
        )

        # No batchnorm here on purpose
        self.generations = tf.nn.sigmoid(
            slim.conv2d_transpose(
                c1, self.image_depth, [5, 5], 2, activation_fn=None,
                padding="SAME",
            )
        )

    def _init_discriminate(self):
        im_mask = tf.tile(
            tf.reshape(self.mask, [self.batch_size, 1, 1, 1]),
            [1, self.image_size, self.image_size, self.image_depth]
        )

        # No batchnorm here on purpose
        input_images = self.real_images * im_mask + self.generations * (1 - im_mask)

        # Convolution 1
        conv1 = slim.conv2d(
            input_images,
            num_outputs=32, kernel_size=[5, 5],
            stride=[2, 2], padding='SAME',
            normalizer_fn=slim.batch_norm,
        )
        level1 = conv1

        # Convolution 2
        conv2 = slim.conv2d(
            level1,
            num_outputs=32, kernel_size=[5, 5],
            stride=[2, 2], padding='SAME',
            normalizer_fn=slim.batch_norm,
        )
        level2 = conv2

        # Level 3 : Fully connected
        level3 = slim.fully_connected(
            slim.flatten(level2),
            100,
        )

        level3 = slim.dropout(level3, 0.5)
        # level3 = tf.concat([slim.flatten(level3), self.label_onehot], axis=1)

        self.discriminate_output = slim.fully_connected(
            level3,
            self.n_classes,
            activation_fn=None
        )

    def _init_losses(self):
        with tf.variable_scope("generator"):
            # generator loss
            self.generator_loss = tf.reduce_mean(
                tf.reshape((1 - self.mask), [self.batch_size, 1]) *
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.discriminate_output,
                    labels=self.label_onehot,
                    name="loss"
                )
            )
            generator_variables = list(filter(
                lambda v: v.name.startswith('dcgan/generate'),
                tf.trainable_variables())
            )
            self.generator_train_step = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.5).minimize(
                self.generator_loss, var_list=generator_variables,
                name="train_step",
            )

        with tf.variable_scope("discriminator"):
            discriminator_labels = tf.reshape(self.mask, [self.batch_size, 1]) * self.label_onehot

            self.discriminator_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.squeeze(self.discriminate_output),
                    labels=discriminator_labels
                ),
                name="loss"
            )
            discriminator_variables = list(filter(
                lambda v: v.name.startswith('dcgan/discriminate'),
                tf.trainable_variables()))

            self.discriminator_train_step = tf.train.AdamOptimizer(LEARNING_RATE, beta1=0.5).minimize(
                self.discriminator_loss, var_list=discriminator_variables,
                name="train_step",
            )


if __name__ == '__main__':
    DCGAN()

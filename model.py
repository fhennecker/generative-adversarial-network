import tensorflow as tf
import tensorflow.contrib.slim as slim


class DCGAN():
    def __init__(self, batch_size=10, n_classes=10, image_size=28):
        with tf.variable_scope("dcgan"):
            self.batch_size = batch_size
            self.n_classes = n_classes
            self.image_size = image_size

            with tf.variable_scope("generate"):
                self._init_generate()
            with tf.variable_scope("discriminate"):
                self._init_discriminate()

        print(self.generations.name)

    def _init_generate(self):
        self.random = tf.placeholder(tf.float32, [self.batch_size, 100])
        self.g_label = tf.placeholder(tf.int32, [self.batch_size])
        label_onehot = tf.one_hot(self.g_label, self.n_classes)
        # label_map is used to append feature maps of 0s and 1s to conv layers
        label_map = tf.reshape(
            label_onehot,
            [self.batch_size, 1, 1, self.n_classes])

        # TODO batch normalise
        h1 = slim.fully_connected(
            tf.concat([self.random, label_onehot], 1),
            512)
        h2 = slim.fully_connected(
            tf.concat([h1, label_onehot], 1),
            128 * 7 * 7)
        h2 = tf.reshape(h2, [self.batch_size, 7, 7, 128])
        h2 = tf.concat(
            [h2, label_map * tf.ones([self.batch_size, 7, 7, self.n_classes])],
            3)

        c1 = slim.conv2d_transpose(h2, 64, [5, 5], 2)
        c1 = tf.concat(
            [c1, label_map * tf.ones([self.batch_size, 14, 14, self.n_classes])],
            3)
        self.generations = slim.conv2d_transpose(c1, 1, [5, 5], 2)

    def _init_discriminate(self):
        self.discriminate_input = tf.placeholder(
            tf.float32,
            [self.batch_size, self.image_size, self.image_size, 1])

        self.discriminate_label = tf.placeholder(tf.int32, [self.batch_size])

        label_onehot = tf.one_hot(self.discriminate_label, self.n_classes)

        label_map = tf.reshape(
            label_onehot,
            [self.batch_size, 1, 1, self.n_classes])

        # Convolution 1
        conv1 = slim.conv2d(
            self.discriminate_input,
            num_outputs=32, kernel_size=[5, 5],
            stride=[2, 2], padding='Valid'
        )
        conv1_shape = conv1.get_shape()[1:3]
        level1_label_map = label_map * tf.ones(
            [self.batch_size, conv1_shape[0], conv1_shape[1], self.n_classes]
        )

        level1 = tf.concat([conv1, level1_label_map], 3)

        # Convolution 2
        conv2 = slim.conv2d(
            level1,
            num_outputs=32, kernel_size=[5, 5],
            stride=[2, 2], padding='Valid'
        )
        conv2_shape = conv2.get_shape()[1:3]
        level2_label_map = label_map * tf.ones(
            [self.batch_size, conv2_shape[0], conv2_shape[1], self.n_classes]
        )

        level2 = tf.concat([conv2, level2_label_map], 3)

        # Level 3
        fc3 = slim.fully_connected(slim.flatten(level2), 200)
        level3 = tf.concat([fc3, label_onehot], 1)

        # Level 4
        level4 = slim.fully_connected(level3, 2)

        self.discriminate_output = tf.nn.softmax(level4)


if __name__ == '__main__':
    DCGAN()

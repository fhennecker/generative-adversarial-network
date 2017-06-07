import tensorflow as tf
import tensorflow.contrib.slim as slim


class DCGAN():
    def __init__(self, batch_size=10, n_classes=10, image_size=28):
        self.batch_size = batch_size
<<<<<<< HEAD
        self.n_classes = n_classes
        self._init_generate()
=======
        self.image_size = image_size
        self.n_classes = n_classes

        self._init_discriminate()
>>>>>>> e9ae4d773b42d102abbed820a4828d46e7bdb1ab

    def _init_generate(self):
        self.random = tf.placeholder(tf.float32, [self.batch_size, 100])
        self.g_label = tf.placeholder(tf.int32, [self.batch_size])
        label_onehot = tf.one_hot(self.g_label, self.n_classes)
        # label_map is used to append feature maps of 0s and 1s to conv layers
        label_map = tf.reshape(label_onehot,
                [self.batch_size, 1, 1, self.n_classes])

        # TODO batch normalise
        h1 = slim.fully_connected(
                tf.concat([self.random, label_onehot], 1),
                512)
        h2 = slim.fully_connected(
                tf.concat([h1, label_onehot], 1),
                128*7*7)
        h2 = tf.reshape(h2, [self.batch_size, 7, 7, 128])
        h2 = tf.concat(
                [h2, label_map*tf.ones([self.batch_size, 7, 7, self.n_classes])],
                3)
        
        c1 = slim.conv2d_transpose(h2, 64, [5, 5], 2)
        c1 = tf.concat(
                [c1, label_map*tf.ones([self.batch_size, 14, 14, self.n_classes])],
                3)
        self.generations = slim.conv2d_transpose(c1, 1, [5, 5], 2)

   def _init_discriminate(self):
        self.discriminate_input = tf.placeholder(tf.float32, 
                [self.batch_size, self.image_size, self.image_size, 1])

        conv1 = slim.conv2d(
            self.discriminate_input,
            num_outputs=32, kernel_size=[5, 5],
            stride=[2, 2], padding='Valid'
        )
        level1 = tf.nn.relu(conv1)

        conv2 = slim.conv2d(
            level1,
            num_outputs=32, kernel_size=[5, 5],
            stride=[2, 2], padding='Valid'
        )
        level2 = tf.nn.relu(conv2)

        level3 = slim.fully_connected(level2, 200)
        level4 = slim.fully_connected(level3, 2)

        self.discriminate_output = tf.nn.softmax(level4)


if __name__ == '__main__':
    DCGAN()

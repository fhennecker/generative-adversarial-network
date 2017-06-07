import tensorflow as tf
import tensorflow.contrib.slim as slim


class DCGAN():
    def __init__(self, batch_size=10, n_classes=10, image_size=28):
        self.batch_size = batch_size
        self.image_size = image_size
        self.n_classes = n_classes

        self._init_discriminate()

    def _init_generate():
        #  self.random = tf.placeholder(tf.float32, [batch_size, 100])
        #  self.label = tf.placeholder(tf.int32, [batch_size])

        #  label_onehot = tf.one_hot(self.label, self.n_classes)
        pass

    def _init_discriminate(self):
        self.discriminate_input = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1])

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

import tensorflow as tf
import tensorflow.contrib.slim as slim


class DCGAN():
    def __init__(self, batch_size=10, n_classes=10):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self._init_generate()

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

    def _init_discriminate():
        pass
    
if __name__ == "__main__":
    DCGAN()
        
        

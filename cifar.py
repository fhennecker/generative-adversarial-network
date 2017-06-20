import pickle
import random
import numpy as np

class CIFAR:
    def __init__(self, path):
        self.databatches = list(map(
            lambda b: self.unpickle(path+'/data_batch_'+b),
            list(map(str, range(1, 6)))))
        self.databatch_len = len(self.databatches[0][b'labels'])

    def batch(self, batch_size):
        data_batch = int(random.choice(range(5)))
        batch_indices = np.random.choice(np.arange(self.databatch_len),
             batch_size, replace=False).astype('int32')

        X = np.reshape(
            self.databatches[data_batch][b'data'][batch_indices],
            [batch_size, 32, 32, 3], 'F').transpose((0, 2, 1, 3))
        y = np.array(self.databatches[data_batch][b'labels'])[batch_indices]
        return X, y

    def unpickle(self, databatch):
        with open(databatch, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

if __name__ == "__main__":
    import cv2
    cifar = CIFAR('./cifar/cifar-10-batches-py/')
    for image, label in zip(*cifar.batch(10)):
        print(label)
        cv2.imshow('Bonsoir', image)
        cv2.waitKey(-1)

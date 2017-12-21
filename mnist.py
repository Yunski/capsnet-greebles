import os
import numpy as np

from sklearn.model_selection import train_test_split

def load_mnist(batch_size, samples_per_epoch=None, is_training=True):
    path = os.path.join('data', 'mnist')
    if is_training:
        train_imgs = open(os.path.join(path, 'train-images-idx3-ubyte'))
        X = np.fromfile(file=train_imgs, dtype=np.uint8)
        X = X[16:].reshape((60000, 28, 28, 1)).astype(np.float32) / 255
        train_labs = open(os.path.join(path, 'train-labels-idx1-ubyte'))
        Y = np.fromfile(file=train_labs, dtype=np.uint8)
        Y = Y[8:].reshape((60000)).astype(np.int32)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=5000)
        num_train_batches = samples_per_epoch // batch_size if samples_per_epoch else len(Y_train) // batch_size
        num_val_batches = len(Y_val) // batch_size
    
        return X_train, X_val, Y_train, Y_val, num_train_batches, num_val_batches 
    else:
        test_imgs = open(os.path.join(path, 't10k-images-idx3-ubyte'))
        X_test = np.fromfile(file=test_imgs, dtype=np.uint8)
        X_test = X_test[16:].reshape((10000, 28, 28, 1)).astype(np.float32) / 255

        test_labs = open(os.path.join(path, 't10k-labels-idx1-ubyte'))
        Y_test = np.fromfile(file=test_labs, dtype=np.uint8)
        Y_test = Y_test[8:].reshape((10000)).astype(np.int32)
    
        num_test_batches = len(Y_test) // batch_size
    
        return X_test, Y_test, num_test_batches


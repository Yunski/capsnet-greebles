import numpy as np
import tensorflow as tf

from mnist import load_mnist
from affnist import load_affnist

def load_data(dataset, batch_size, is_training=True, samples_per_epoch=None):
    if dataset == 'mnist':
        return load_mnist(batch_size, samples_per_epoch, is_training)
    elif dataset == 'affnist':
        return load_affnist(batch_size, samples_per_epoch, is_training)
    else:
        return None

def get_train_batch(dataset, batch_size, num_threads, min_after_dequeue=5000, samples_per_epoch=None):
    if dataset == 'mnist':
        X_train, X_val, Y_train, Y_val, num_train_batches, num_val_batches = load_mnist(batch_size, samples_per_epoch=samples_per_epoch)
    elif dataset == 'affnist':
        X_train, X_val, Y_train, Y_val, num_train_batches, num_val_batches = load_affnist(batch_size, samples_per_epoch=samples_per_epoch)
    data_queues = tf.train.slice_input_producer([X_train, Y_train])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=min_after_dequeue + (num_threads + 1) * batch_size,
                                  min_after_dequeue=min_after_dequeue,
                                  allow_smaller_final_batch=False)

    return X, Y


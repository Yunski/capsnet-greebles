import os
import re
import numpy as np
import tensorflow as tf

from mnist import load_mnist
from affnist import load_affnist, read_affnist_tfrecord
from smallnorb import load_norb, read_norb_tfrecord
from greebles import load_greebles, read_greebles_tfrecord

def load_data(dataset, batch_size, is_training=True, samples_per_epoch=None, use_val_only=True):
    if samples_per_epoch == 0:
        samples_per_epoch = None
    if dataset == 'mnist':
        return load_mnist(batch_size, samples_per_epoch, is_training, use_val_only)
    elif dataset == 'affnist':
        return load_affnist(batch_size, samples_per_epoch, is_training)
    elif dataset == 'smallnorb':
        return load_norb(batch_size, samples_per_epoch, is_training)
    elif dataset == 'greebles':
        return load_greebles(batch_size, samples_per_epoch, is_training)
    else:
        raise ValueError("{} is not an available dataset".format(dataset))


def get_train_batch(dataset, batch_size, num_threads, min_after_dequeue=5000, samples_per_epoch=None):
    if samples_per_epoch == 0:
        samples_per_epoch = None
    if dataset == 'mnist':
        X_train, X_val, Y_train, Y_val, num_train_batches, num_val_batches = load_mnist(batch_size, samples_per_epoch=samples_per_epoch)
        data_queues = tf.train.slice_input_producer([X_train, Y_train])
    elif dataset in set(['affnist', 'smallnorb', 'greebles']):
        CHUNK_RE = re.compile(r"train-\d+\.tfrecords")
        data_dir = "data/{}".format(dataset)
        chunk_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if CHUNK_RE.match(fname)]
        if dataset == 'affnist':
            X_train, Y_train = read_affnist_tfrecord(chunk_files)
        elif dataset == 'smallnorb':
            X_train, Y_train = read_norb_tfrecord(chunk_files)
        else:
            X_train, Y_train = read_greebles_tfrecord(chunk_files)
        
        if dataset == 'smallnorb':
            X_train = tf.image.random_brightness(X_train, max_delta=32. / 255.)
            X_train = tf.image.random_contrast(X_train, lower=0.5, upper=1.5)
        if dataset != 'affnist':
            X_train = tf.image.resize_images(X_train, [48, 48])
            X_train = tf.random_crop(X_train, [32, 32, 1])
        X_train = X_train / 255
        data_queues = [X_train, Y_train]
    else:
        raise ValueError("{} is not an available dataset".format(dataset))

    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=min_after_dequeue + (num_threads + 1) * batch_size,
                                  min_after_dequeue=min_after_dequeue,
                                  allow_smaller_final_batch=False)
    return X, Y


def get_test_batch(dataset, batch_size, num_threads, min_after_dequeue=5000, samples_per_epoch=None):
    if dataset in set(['affnist', 'smallnorb', 'greebles']):
        CHUNK_RE = re.compile(r"test-\d+\.tfrecords")
        data_dir = "data/{}".format(dataset)
        chunk_files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if CHUNK_RE.match(fname)]

        if dataset == 'affnist':
            X_test, Y_test = read_affnist_tfrecord(chunk_files)
        elif dataset == 'smallnorb':
            X_test, Y_test = read_norb_tfrecord(chunk_files)
        else:
            X_test, Y_test = read_greebles_tfrecord(chunk_files)
 
        if dataset != 'affnist':
            X_test = tf.image.resize_images(X_test, [48, 48])
            X_test = tf.slice(X_test, [8, 8, 0], [32, 32, 1])
        X_test = X_test / 255
        data_queues = [X_test, Y_test]
    else:
        raise ValueError("{} is not an available dataset".format(dataset))

    X, Y = tf.train.batch(data_queues, num_threads=num_threads,
                          batch_size=batch_size,
                          capacity=min_after_dequeue + (num_threads + 1) * batch_size,
                          allow_smaller_final_batch=False)

    return X, Y


def get_dataset_values(dataset, batch_size, is_training=True):
    if dataset == 'mnist':
        input_shape = (batch_size, 28, 28, 1)
        num_classes = 10
        use_test_queue = False
    elif dataset == 'affnist':
        input_shape = (batch_size, 40, 40, 1)
        num_classes = 10
        use_test_queue = True
    elif dataset == 'smallnorb':
        input_shape = (batch_size, 32, 32, 1)
        num_classes = 5
        use_test_queue = True
    elif dataset == 'greebles':
        input_shape = (batch_size, 32, 32, 1)
        num_classes = 5
        use_test_queue = True
    else:
        raise ValueError("{} is not an available dataset".format(dataset))
    if is_training:
        return input_shape, num_classes
    else:
        return input_shape, num_classes, use_test_queue


def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = None
        if callable(initializer):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        else:
            var = tf.get_variable(name, initializer=initializer)
    return var

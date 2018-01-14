import argparse
import glob
import os
import random
import re
import scipy.io
import sys
import re
import time
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def convertPngsToNPY(visualize=False):
    print("Converting pngs to numpy array file...")
    path = os.path.join("data", "affnist")

    train_data = loadmat(os.path.join(path, 'training.mat'))
    train_images = train_data['affNISTdata']['image'].transpose()
    train_images = train_images.reshape((train_images.shape[0], 40, 40, 1)).astype(np.float64)
    train_images = np.array([random_shift(image) for image in train_images]) 
    train_labels = train_data['affNISTdata']['label_int']
    train_labels = train_labels.reshape((train_labels.shape[0])).astype(np.int64)

    print("Finished loading training.mat")
    
    val_data = loadmat(os.path.join(path, 'validation.mat'))
              
    val_images = val_data['affNISTdata']['image'].transpose()
    val_images = val_images.reshape((val_images.shape[0], 40, 40, 1)).astype(np.float64)
    val_images = np.array([random_shift(image) for image in val_images])
    val_labels = val_data['affNISTdata']['label_int']
    val_labels = val_labels.reshape((val_labels.shape[0])).astype(np.int64)

    print("Finished loading validation.mat")

    test_data = loadmat(os.path.join(path, 'test.mat'))
    test_images = test_data['affNISTdata']['image'].transpose()
    test_images = test_images.reshape((test_images.shape[0], 40, 40, 1)).astype(np.float64)
    test_labels = test_data['affNISTdata']['label_int']
    test_labels = test_labels.reshape((test_labels.shape[0])).astype(np.int64)
        
    print("Finished loading test.mat")

    train_images, train_labels = shuffle(train_images, train_labels)
    val_images, val_labels = shuffle(val_images, val_labels)
    test_images, test_labels = shuffle(test_images, test_labels)

    if visualize:
        num_vis_images = 20
        np.save("data/affnist/vis-train-images.npy", train_images[:num_vis_images])
        np.save("data/affnist/vis-train-labs.npy", train_labels[:num_vis_images])
        np.save("data/affnist/vis-test-images.npy", test_images[:num_vis_images])
        np.save("data/affnist/vis-test-labs.npy", test_labels[:num_vis_images])
        print("Successfully saved visualization arrays.")
    else:
        np.save("data/affnist/train-images.npy", train_images)
        np.save("data/affnist/train-labs.npy", train_labels)
        np.save("data/affnist/val-images.npy", val_images)
        np.save("data/affnist/val-labs.npy", val_labels)
        np.save("data/affnist/test-images.npy", test_images)
        np.save("data/affnist/test-labs.npy", test_labels)
        print("Successfully converted pngs.")


def write_data_to_tfrecord(is_training=True, chunkify=False):
    """
    Adapted from https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow/blob/master/data/smallNORB.py
    """
    kind = "train" if is_training else "test"
    print("Start writing afffnist {} data.".format(kind))

    start = time.time()
    if is_training:
        images = np.load("data/affnist/train-images.npy")
        labels = np.load("data/affnist/train-labs.npy")
    else:
        images = np.load("data/affnist/test-images.npy")
        labels = np.load("data/affnist/test-labs.npy")

    total_num_images = len(images)
    CHUNK = total_num_images // 10  # create 10 chunks

    for j in range(total_num_images // CHUNK if chunkify else 1):
        num_images = CHUNK if chunkify else total_num_images

        print("Start filling chunk {}.".format(j))

        perm = np.random.permutation(num_images)
        images = images[perm]
        labels = labels[perm]

        writer = tf.python_io.TFRecordWriter("data/affnist/{}-{}.tfrecords".format(kind, j))
        for i in range(num_images):
            img = images[i].tostring()
            lab = labels[i].astype(np.int64)
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[lab])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))
            }))
            writer.write(example.SerializeToString())
        writer.close()

    print("Done writing {} data. Total time: {:.4f}s.".format(kind, time.time() - start))


def tfrecord():
    write_data_to_tfrecord(is_training=True, chunkify=False)
    write_data_to_tfrecord(is_training=False, chunkify=False)


def read_affnist_tfrecord(filenames, num_epochs=None):
    """
    from https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow/blob/master/data/smallNORB.py
    """

    assert isinstance(filenames, list)

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.float64)
    img = tf.reshape(img, [40, 40, 1])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return img, label


def load_affnist(batch_size, samples_per_epoch=None, is_training=True):
    if is_training:
        train_labels = np.load("data/affnist/train-labs.npy")
        val_images = np.load("data/affnist/val-images.npy")
        val_labels = np.load("data/affnist/val-labs.npy")

        val_images = val_images / 255

        num_train_batches = samples_per_epoch // batch_size if samples_per_epoch else len(train_labels) // batch_size
        num_val_batches = len(val_labels) // batch_size 
        # do not provide training data here
        return [], val_images, [], val_labels, num_train_batches, num_val_batches 
    else:
        test_labels = np.load("data/affnist/test-labs.npy")
        num_test_batches = len(test_labels) // batch_size
        # do not provide test data here
        return [], [], num_test_batches


def test(is_training=True):
    if is_training:
        CHUNK_RE = re.compile(r"train-\d+\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test-\d+\.tfrecords")

    processed_dir = 'data/affnist'
    chunk_files = [os.path.join(processed_dir, fname)
                   for fname in os.listdir(processed_dir)
                   if CHUNK_RE.match(fname)]
    image, label = read_affnist_tfrecord(chunk_files)

    batch_size = 8
    x, y = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32, allow_smaller_final_batch=False)
    print("x shape: {}, y shape: {}".format(x.get_shape(), y.get_shape()))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(10):
            val, l = sess.run([x, y])
            print(val, l)

        coord.request_stop()
        coord.join(threads)

    print("Successfully completed test.")


def random_shift(image):
    return tf.contrib.keras.preprocessing.image.random_shift(image, 0.35, 0.35)


def loadmat(filename):
    '''
    this function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Affnist Data Writer')
    parser.add_argument('-f', '--force', action='store_true') 
    parser.add_argument('-t', '--test', action='store_true') 
    parser.add_argument('-v', '--visualize', action='store_true') 
    args = parser.parse_args()

    train_imgs_file = "data/affnist/train-images.npy"
    train_labs_file = "data/affnist/train-labs.npy"
    val_imgs_file = "data/affnist/val-images.npy"
    val_labs_file = "data/affnist/val-labs.npy"
    test_imgs_file = "data/affnist/test-images.npy"
    test_labs_file = "data/affnist/test-labs.npy"

    if args.test:
        test()
    elif args.visualize:
        convertPngsToNPY(visualize=True)
    else: 
        if args.force or (not os.path.isfile(train_imgs_file) or \
            not os.path.isfile(train_labs_file) or \
            not os.path.isfile(val_imgs_file) or \
            not os.path.isfile(val_labs_file) or \
            not os.path.isfile(test_imgs_file) or \
            not os.path.isfile(test_labs_file)):
            for filepath in glob.glob("data/affnist/*.npy"):
                os.remove(filepath)
            convertPngsToNPY()
        for filepath in glob.glob("data/affnist/*.tfrecords"):
            os.remove(filepath)
        tfrecord()


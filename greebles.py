import argparse
import glob
import os
import random
import re
import sys
import re
import time
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def convertPngsToNPY(n, visualize=False):
    print("Converting pngs to numpy array file...")
    path = os.path.join("data", "greebles")
    train_path = os.path.join(path, "train/*.png")
    test_path = os.path.join(path, "test/*.png")
    train_image_paths = glob.glob(train_path)
    test_image_paths = glob.glob(test_path)
    num_train_images = len(train_image_paths) if not visualize else 20
    num_test_images = len(test_image_paths) if not visualize else 20
    random.shuffle(train_image_paths)
    random.shuffle(test_image_paths)

    train_images = np.zeros((num_train_images, n, n, 1))
    train_labels = np.zeros((num_train_images))

    test_images = np.zeros((num_test_images, n, n, 1))
    test_labels = np.zeros((num_test_images))

    for i, image_path in enumerate(train_image_paths):
        if i == num_train_images:
            break
        img = Image.open(image_path).convert('L')
        train_images[i] = np.array(img).reshape(n, n, 1)
        train_labels[i] = int(re.search('\d+', image_path).group()) - 1

    for i, image_path in enumerate(test_image_paths):
        if i == num_test_images:
            break
        img = Image.open(image_path).convert('L')
        test_images[i] = np.array(img).reshape(n, n, 1)
        test_labels[i] = int(re.search('\d+', image_path).group()) - 1

    train_images, train_labels = shuffle(train_images, train_labels)
    test_images, test_labels = shuffle(test_images, test_labels)

    if visualize:
        np.save("data/greebles/vis-train-images.npy", train_images)
        np.save("data/greebles/vis-train-labs.npy", train_labels)
        np.save("data/greebles/vis-test-images.npy", test_images)
        np.save("data/greebles/vis-test-labs.npy", test_labels)
        print("Successfully saved visualization arrays.")
    else:
        np.save("data/greebles/train-images.npy", train_images)
        np.save("data/greebles/train-labs.npy", train_labels)
        np.save("data/greebles/test-images.npy", test_images)
        np.save("data/greebles/test-labs.npy", test_labels)
        print("Successfully converted pngs.")


def write_data_to_tfrecord(is_training=True, chunkify=False):
    """
    Adapted from https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow/blob/master/data/smallNORB.py
    """
    kind = "train" if is_training else "test"
    print("Start writing greebles {} data.".format(kind))

    start = time.time()
    if is_training:
        images = np.load("data/greebles/train-images.npy")
        labels = np.load("data/greebles/train-labs.npy")
    else:
        images = np.load("data/greebles/test-images.npy")
        labels = np.load("data/greebles/test-labs.npy")

    total_num_images = len(images)
    CHUNK = total_num_images // 10  # create 10 chunks

    for j in range(total_num_images // CHUNK if chunkify else 1):
        num_images = CHUNK if chunkify else total_num_images

        print("Start filling chunk {}.".format(j))

        perm = np.random.permutation(num_images)
        images = images[perm]
        labels = labels[perm]

        writer = tf.python_io.TFRecordWriter("data/greebles/{}-{}.tfrecords".format(kind, j))
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


def read_greebles_tfrecord(filenames, n=96):
    """
    from https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow/blob/master/data/smallNORB.py
    """

    assert isinstance(filenames, list)

    filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.float64)
    img = tf.reshape(img, [n, n, 1])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return img, label


def load_greebles(batch_size, samples_per_epoch=None, is_training=True):
    if is_training:
        num_train_batches = samples_per_epoch // batch_size if samples_per_epoch else 16000 // batch_size
        # do not provide training or validation data here
        return [], [], [], [], num_train_batches, 0
    else:
        num_test_batches = 8000 // batch_size
        # do not provide test data here
        return [], [], num_test_batches


def test(is_training=True):
    if is_training:
        CHUNK_RE = re.compile(r"train-\d+\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test-\d+\.tfrecords")

    processed_dir = 'data/greebles'
    chunk_files = [os.path.join(processed_dir, fname)
                   for fname in os.listdir(processed_dir)
                   if CHUNK_RE.match(fname)]
    image, label = read_greebles_tfrecord(chunk_files)
    # image = tf.image.random_brightness(image, max_delta=32. / 255.)
    # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    image = tf.image.resize_images(image, [48, 48])

    """
    params_shape = [image.get_shape()[-1]]
    beta = tf.get_variable(
        'beta', params_shape, tf.float32,
        initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable(
        'gamma', params_shape, tf.float32,
        initializer=tf.constant_initializer(1.0, tf.float32))
    mean, variance = tf.nn.moments(image, [0, 1, 2])
    image = tf.nn.batch_normalization(image, mean, variance, beta, gamma, 0.001)
    """

    image = tf.random_crop(image, [32, 32, 1])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greebles Data Writer')
    parser.add_argument('-n', help='image dimensions', dest='n', type=int, default=96) 
    parser.add_argument('-f', '--force', action='store_true') 
    parser.add_argument('-t', '--test', action='store_true') 
    parser.add_argument('-v', '--visualize', action='store_true') 
    args = parser.parse_args()

    train_imgs_file = "data/greebles/train-images.npy"
    train_labs_file = "data/greebles/train-labs.npy"
    test_imgs_file = "data/greebles/test-images.npy"
    test_labs_file = "data/greebles/test-labs.npy"

    if args.test:
        test()
    elif args.visualize:
        convertPngsToNPY(args.n, visualize=True)
    else: 
        if args.force or (not os.path.isfile(train_imgs_file) or \
            not os.path.isfile(train_labs_file) or \
            not os.path.isfile(test_imgs_file) or \
            not os.path.isfile(test_labs_file)):
            convertPngsToNPY(args.n)
        for filepath in glob.glob("data/greebles/*.tfrecords"):
            os.remove(filepath)
        tfrecord()


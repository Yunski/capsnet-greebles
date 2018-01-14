import argparse
import os
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(10234)

def plot_imgs(inputs, labels):
    """
    Adapted from https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow/blob/master/data/smallNORB.py
    """
    fig = plt.figure()
    plt.axis('off')
    r = np.floor(np.sqrt(len(inputs))).astype(int)
    for i in range(r**2):
        size = int(np.sqrt(inputs[i].shape[0]))
        sample = inputs[i].flatten().reshape(size, size)
        a = fig.add_subplot(r, r, i + 1)
        a.set_title(labels[i])
        a.axis('off')
        a.imshow(sample, cmap='gray')
    plt.show()


def write_data_to_tfrecord(is_training=True, chunkify=False):
    """
    Adapted from https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow/blob/master/data/smallNORB.py
    """

    kind = "train" if is_training else "test"
    print("Start writing smallnorb {} data.".format(kind))
    CHUNK = 24300 * 2 // 10  # create 10 chunks

    start = time.time()
    if is_training:
        fid_images = open('data/smallnorb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat', 'rb')
        fid_labels = open('data/smallnorb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat', 'rb')
    else:
        fid_images = open('data/smallnorb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat', 'rb')
        fid_labels = open('data/smallnorb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat', 'rb')

    for i in range(6):
        a = fid_images.read(4) # header

    total_num_images = 24300 * 2

    for j in range(total_num_images // CHUNK if chunkify else 1):
        num_images = CHUNK if chunkify else total_num_images  # 24300 * 2
        images = np.zeros((num_images, 96 * 96))
        for idx in range(num_images):
            temp = fid_images.read(96 * 96)
            images[idx, :] = np.fromstring(temp, 'uint8')
        for i in range(5):
            a = fid_labels.read(4) # header
        labels = np.fromstring(fid_labels.read(num_images * np.dtype('int32').itemsize), 'int32')
        labels = np.repeat(labels, 2)

        print("Start filling chunk {}.".format(j))

        perm = np.random.permutation(num_images)
        images = images[perm]
        labels = labels[perm]

        # if j == 0:
        #     plot_imgs(images[:9], labels[:9])

        writer = tf.python_io.TFRecordWriter("data/smallnorb/{}-{}.tfrecords".format(kind, j))
        for i in range(num_images):
            img = images[i, :].tobytes()
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


def read_norb_tfrecord(filenames, num_epochs=None):
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
    img = tf.reshape(img, [96, 96, 1])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return img, label


def load_norb(batch_size, samples_per_epoch=None, is_training=True):
    if is_training:
        num_train_batches = samples_per_epoch // batch_size if samples_per_epoch else 24300 * 2 // batch_size
        # do not provide training or validation data here
        return [], [], [], [], num_train_batches, 0
    else:
        num_test_batches = 24300 * 2 // batch_size
        # do not provide test data here
        return [], [], num_test_batches


def test(is_training=True):
    if is_training:
        CHUNK_RE = re.compile(r"train-\d+\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test-\d+\.tfrecords")

    processed_dir = 'data/smallnorb'
    chunk_files = [os.path.join(processed_dir, fname)
                   for fname in os.listdir(processed_dir)
                   if CHUNK_RE.match(fname)]
    image, label = read_norb_tfrecord(chunk_files)

    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    image = tf.image.resize_images(image, [48, 48])

    """Batch Norm"""
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
    image = image / 255
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

    print('Successfully completed test.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smallnorb Data Writer")
    parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        test()
    else:
        tfrecord()

# VGG-like CNN
#
# Can't use original VGG because
# (1) images too small 
# (2) deeper networks difficult to train without residual connections
# 
# Architecture:
# Block 1: two 3x3 convolutions (64 channels), one 2x2 max-pool
# Block 2: two 3x3 convolutions (128 channels), one 2x2 max-pool
# Block 3: two 3x3 convolutions (256 channels), one 2x2 max-pool
# two fully-connected layers (one with 512 channels, one with 10 channels)
# softmax layer
#
# total: 6 convolutional layers, 3 max-pool layers, 2 fully-connected layers


import tensorflow as tf

from config import cfg
from utils import get_train_batch, get_test_batch, variable_on_cpu


class VGGNet(object):
    def __init__(self, input_shape, num_classes, is_training=True, use_test_queue=False):
        self.input_shape = input_shape
        self.name = "vggnet"
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_train_batch(cfg.dataset, cfg.batch_size, cfg.num_threads, samples_per_epoch=cfg.samples_per_epoch)
                self.inference(self.X, num_classes)
                self.loss()
                self._summary()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=0.01)
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                if use_test_queue:
                    self.X, self.labels = get_test_batch(cfg.dataset, cfg.test_batch_size, cfg.num_threads)
                else:
                    self.X = tf.placeholder(tf.float32, shape=self.input_shape)
                    self.labels = tf.placeholder(tf.int32, shape=(self.input_shape[0],))
                self.inference(self.X, num_classes, keep_prob=1.0)
                self.loss()
                self.error()

    def inference(self, inputs, num_classes, keep_prob=0.5):
        def conv_3x3_with_relu(x, channels):
            kernel = variable_on_cpu('weights', shape=[3, 3, x.shape[-1].value, channels],
                                     initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = variable_on_cpu('biases', [channels], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv_with_activation = tf.nn.relu(pre_activation, name=scope.name)
            return conv_with_activation

        def pool_2x2(x):
            pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            return pool

        def fully_connected(x, channels):
            weights = variable_on_cpu('weights',
                                      shape=[x.shape[1].value, channels],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = variable_on_cpu('biases', [channels], tf.constant_initializer(0.0))
            fc = tf.nn.relu(tf.matmul(x, weights) + biases, name=scope.name)
            return fc

        def dropout(x):
            dropout = tf.nn.dropout(x, keep_prob)
            return dropout
            
        # block 1: two 3x3 conv filters, one 2x2 max-pool, 64 channels
        with tf.variable_scope('conv1') as scope:
            conv1 = conv_3x3_with_relu(inputs, 64)
        
        with tf.variable_scope('conv2') as scope:
            conv2 = conv_3x3_with_relu(conv1, 64)
        
        with tf.name_scope('pool1') as scope:
            pool1 = pool_2x2(conv2)
        
        # block 2: two 3x3 conv filters, one 2x2 max-pool, 128 channels
        with tf.variable_scope('conv3') as scope:
            conv3 = conv_3x3_with_relu(pool1, 128)
        
        with tf.variable_scope('conv4') as scope:
            conv4 = conv_3x3_with_relu(conv3, 128)
        
        with tf.name_scope('pool2') as scope:
            pool2 = pool_2x2(conv4)
        
        # block 3: two 3x3 conv filters, one 2x2 max-pool, 256 channels
        with tf.variable_scope('conv5') as scope:
            conv5 = conv_3x3_with_relu(pool2, 256)
        
        with tf.variable_scope('conv6') as scope:
            conv6 = conv_3x3_with_relu(conv5, 256)
        
        with tf.name_scope('pool3') as scope:
            pool3 = pool_2x2(conv6)

        # two fully-connected layers, with dropout after first
        with tf.variable_scope('fc1') as scope:
            reshape = tf.reshape(pool3, [pool3.shape[0].value, -1])
            fc1 = fully_connected(reshape, 512)

        with tf.variable_scope('dropout') as scope:
            dropout = dropout(fc1)
        
        with tf.variable_scope('fc2') as scope:
            fc2 = fully_connected(dropout, num_classes)

        # final softmax layer
        with tf.name_scope('softmax') as scope:
            logits = tf.nn.softmax(fc2)

        self.logits = logits

    def loss(self):
        # regularization code adapted from https://stackoverflow.com/a/38466108
        # beta = 0.01
        # regularizer = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "biases" not in v.name])
        self.total_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.labels
            )) # + beta * regularizer

    def error(self):
        self.predictions = tf.to_int32(tf.argmax(self.logits, axis=1))
        errors = tf.not_equal(tf.to_int32(self.labels), self.predictions)
        self.error_rate = tf.reduce_mean(tf.cast(errors, tf.float32))

    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        self.train_summary = tf.summary.merge(train_summary)
        self.error()

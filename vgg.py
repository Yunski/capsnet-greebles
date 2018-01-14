# smaller VGG-like CNN for training bigger VGG-like model
# 
# Architecture:
# Block 1: one 3x3 convolution (64 channels), one 2x2 max-pool
# Block 2: one 3x3 convolution (128 channels), one 2x2 max-pool
# Block 3: one 3x3 convolution (256 channels), one 2x2 max-pool
# Block 4: one 3x3 convolution (512 channels), one 2x2 max-pool
# two fully-connected layers (one with 512 channels, one with 10 channels)
# softmax layer
#
# total: 4 convolutional layers, 4 max-pool layers, 2 fully-connected layers


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
                self.X, self.labels = get_train_batch(cfg.dataset, cfg.batch_size, cfg.num_threads,
                                                      samples_per_epoch=cfg.samples_per_epoch)
                self.inference(self.X, num_classes)
                self.loss()
                self._summary()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                learning_rate = 1e-4
                self.optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1)
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                if use_test_queue:
                    self.X, self.labels = get_test_batch(cfg.dataset, cfg.batch_size, cfg.num_threads,
                                                         samples_per_epoch=cfg.samples_per_epoch)
                else:
                    self.X = tf.placeholder(tf.float32, shape=self.input_shape)
                    self.labels = tf.placeholder(tf.int32, shape=(self.input_shape[0],))
                    self.inference(self.X, keep_prob=1.0)
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
            reshape = tf.reshape(x, [x.shape[0].value, -1])
            weights = variable_on_cpu('weights',
                                      shape=[reshape.shape[1].value, channels],
                                      initializer=tf.contrib.layers.xavier_initializer())
            biases = variable_on_cpu('biases', [channels], tf.constant_initializer(0.0))
            fc = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            return fc

        # block 1: three 3x3 conv filters, one 2x2 max-pool, 64 channels
        with tf.variable_scope('conv1') as scope:
            conv1 = conv_3x3_with_relu(inputs, 64)
        
        with tf.variable_scope('conv2') as scope:
            conv2 = conv_3x3_with_relu(conv1, 64)
        
        with tf.name_scope('pool1') as scope:
            pool1 = pool_2x2(conv2)
        
        # block 2: three 3x3 conv filters, one 2x2 max-pool, 128 channels
        with tf.variable_scope('conv3') as scope:
            conv3 = conv_3x3_with_relu(pool1, 128)
        
        with tf.variable_scope('conv4') as scope:
            conv4 = conv_3x3_with_relu(conv3, 128)
        
        with tf.name_scope('pool2') as scope:
            pool2 = pool_2x2(conv4)
        
        # block 3: three 3x3 conv filters, one 2x2 max-pool, 256 channels
        with tf.variable_scope('conv5') as scope:
            conv5 = conv_3x3_with_relu(pool2, 256)
        
        with tf.variable_scope('conv6') as scope:
            conv6 = conv_3x3_with_relu(conv5, 256)
        	
        '''
        with tf.variable_scope('conv7') as scope:
            conv7 = conv_3x3_with_relu(conv6, 256)
        '''

        with tf.name_scope('pool3') as scope:
            pool3 = pool_2x2(conv6)

        '''
        # block 4: three 3x3 conv filters, one 2x2 max-pool, 512 channels
        with tf.variable_scope('conv8') as scope:
            conv8 = conv_3x3_with_relu(pool3, 512)
        
        with tf.variable_scope('conv9') as scope:
            conv9 = conv_3x3_with_relu(conv8, 512)

        with tf.variable_scope('conv10') as scope:
            conv10 = conv_3x3_with_relu(conv9, 512)
        
        with tf.name_scope('pool3') as scope:
            pool4 = pool_2x2(conv8)
        '''
        # two fully-connected layers
        with tf.variable_scope('fc1') as scope:
            fc1 = fully_connected(pool3, 512)
        
        with tf.variable_scope('fc2') as scope:
            fc2 = fully_connected(fc1, num_classes)

        # final softmax layer
        with tf.name_scope('softmax') as scope:
            logits = tf.nn.softmax(fc2)

        self.logits = logits

    def loss(self):
        self.total_loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits,
                labels=self.labels
            )
        )

    def error(self):
        self.predictions = tf.to_int32(tf.argmax(self.logits, axis=1))
        errors = tf.not_equal(tf.to_int32(self.labels), self.predictions)
        self.error_rate = tf.reduce_mean(tf.cast(errors, tf.float32))

    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        self.train_summary = tf.summary.merge(train_summary)
        self.error()

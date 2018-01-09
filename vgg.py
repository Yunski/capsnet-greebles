# VGG-like CNN for small-image classification
# Cannot use original VGG-16 architecture because input images too small:
# We leave off the fifth (final) block of convolution/pooling
#
# Architecture:
# Block 1: two 3x3 convolutions (64 channels), one 2x2 max-pool
# Block 2: two 3x3 convolutions (128 channels), one 2x2 max-pool
# Block 3: three 3x3 convolutions (256 channels), one 2x2 max-pool
# Block 4: three 3x3 convolutions (512 channels), one 2x2 max-pool
# two fully-connected layers (one with 512 channels, one with 10 channels)
# softmax layer
#
# total: 12 convolutional layers, 4 max-pool layers, 2 fully-connected layers


import tensorflow as tf

from config import cfg
from utils import get_train_batch, variable_on_cpu


class VGGNet(object):
    def __init__(self, input_shape, is_training=True, use_test_queue=False):
        self.input_shape = input_shape
        self.name = "vggnet"
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_train_batch(cfg.dataset, cfg.batch_size, cfg.num_threads,
                                                      samples_per_epoch=cfg.samples_per_epoch)
                self.inference(self.X)
                self.loss()
                self._summary()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                learning_rate = 1e-3 # decrease by factor of 10 if val error stops decreasing
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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

    def inference(self, inputs, keep_prob=0.5):
        def conv_3x3_with_relu(x, channels, 
                               kernel_init=tf.contrib.layers.xavier_initializer(), 
                               biases_init=tf.constant_initializer(0.0)):
            kernel = variable_on_cpu('weights', shape=[3, 3, x.shape[-1].value, channels],
                                     initializer=kernel_init)
            conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
            biases = variable_on_cpu('biases', [channels], 
                     initializer=biases_init)
            pre_activation = tf.nn.bias_add(conv, biases)
            conv_with_activation = tf.nn.relu(pre_activation, name=scope.name)
            return conv_with_activation

        def pool_2x2(x):
            pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            return pool

        def fully_connected(x, channels,
                            kernel_init=tf.contrib.layers.xavier_initializer(), 
                            biases_init=tf.constant_initializer(0.0)):
            reshape = tf.reshape(x, [x.shape[0].value, -1])
            weights = variable_on_cpu('weights',
                                      shape=[reshape.shape[1].value, channels],
                                      initializer=kernel_init)
            biases = variable_on_cpu('biases', [channels], biases_init)
            fc = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            return fc

        # grab weights from VGGSmallNet
        small_ckpt_state = tf.train.get_checkpoint_state('logs/vggsmallnet/' + cfg.dataset)
        reader = tf.train.NewCheckpointReader(small_ckpt_state.model_checkpoint_path)

        # block 1: three 3x3 conv filters, one 2x2 max-pool, 64 channels
        with tf.variable_scope('conv1') as scope:
            kernel_init = tf.constant(reader.get_tensor('conv1/weights'))
            biases_init = tf.constant(reader.get_tensor('conv1/biases'))
            conv1 = conv_3x3_with_relu(inputs, 64, kernel_init=kernel_init, biases_init=biases_init)
        
        with tf.variable_scope('conv2') as scope:
            conv2 = conv_3x3_with_relu(conv1, 64)
       
        with tf.name_scope('pool1') as scope:
            pool1 = pool_2x2(conv2)

        # block 2: three 3x3 conv filters, one 2x2 max-pool, 128 channels
        with tf.variable_scope('conv3') as scope:
            kernel_init = tf.constant(reader.get_tensor('conv3/weights'))
            biases_init = tf.constant(reader.get_tensor('conv3/biases'))
            conv3 = conv_3x3_with_relu(pool1, 128, kernel_init=kernel_init, biases_init=biases_init)
        
        with tf.variable_scope('conv4') as scope:
            conv4 = conv_3x3_with_relu(conv3, 128)
        
        with tf.name_scope('pool2') as scope:
            pool2 = pool_2x2(conv4)

        # block 3: three 3x3 conv filters, one 2x2 max-pool, 256 channels
        with tf.variable_scope('conv5') as scope:
            kernel_init = tf.constant(reader.get_tensor('conv5/weights'))
            biases_init = tf.constant(reader.get_tensor('conv5/biases'))
            conv5 = conv_3x3_with_relu(pool2, 256, kernel_init=kernel_init, biases_init=biases_init)
        
        with tf.variable_scope('conv6') as scope:
            conv6 = conv_3x3_with_relu(conv5, 256)

        with tf.variable_scope('conv7') as scope:
            conv7 = conv_3x3_with_relu(conv6, 256)
        
        with tf.name_scope('pool3') as scope:
            pool3 = pool_2x2(conv7)

        # block 4: three 3x3 conv filters, one 2x2 max-pool, 512 channels
        with tf.variable_scope('conv8') as scope:
            kernel_init = tf.constant(reader.get_tensor('conv8/weights'))
            biases_init = tf.constant(reader.get_tensor('conv8/biases'))
            conv8 = conv_3x3_with_relu(pool3, 512)
        
        with tf.variable_scope('conv9') as scope:
            conv9 = conv_3x3_with_relu(conv8, 512)

        with tf.variable_scope('conv10') as scope:
            conv10 = conv_3x3_with_relu(conv9, 512)
        
        with tf.name_scope('pool3') as scope:
            pool4 = pool_2x2(conv10)

        # two fully-connected layers
        with tf.variable_scope('fc1') as scope:
            kernel_init = tf.constant(reader.get_tensor('fc1/weights'))
            biases_init = tf.constant(reader.get_tensor('fc1/biases'))
            fc1 = fully_connected(pool4, 512, kernel_init=kernel_init, biases_init=biases_init)

        with tf.variable_scope('fc2') as scope:
            kernel_init = tf.constant(reader.get_tensor('fc2/weights'))
            biases_init = tf.constant(reader.get_tensor('fc2/biases'))
            fc2 = fully_connected(fc1, 10, kernel_init=kernel_init, biases_init=biases_init)

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

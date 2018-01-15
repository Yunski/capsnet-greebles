import tensorflow as tf

from config import cfg
from utils import get_train_batch, get_test_batch, variable_on_cpu

class resnet(object):
    def __init__(self, input_shape, num_classes, is_training=True, use_test_queue=False):
        self.input_shape = input_shape
        self.name = "resnet"
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_train_batch(cfg.dataset, cfg.batch_size, cfg.num_threads, samples_per_epoch=cfg.samples_per_epoch)
                self.inference(self.X, num_classes)
                self.loss()
                self._summary()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(epsilon=0.1)
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
        nodes = []

        # One convolutional layer with maxpooling before beginning residual learning.
        with tf.variable_scope('conv0') as scope:
            filter_size = 7
            stride = 1
            num_filters = 64
            conv_0 = self.conv_layer(inputs, filter_size, filter_size, inputs.shape[-1], num_filters, stride)
            activation_0 = tf.nn.relu(conv_0, name=scope.name)
        with tf.variable_scope('pool0') as scope:
            pool_0 = tf.nn.max_pool(activation_0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)
            nodes.append(pool_0)

        # Begin residual layers. In ResNet, we have a num_layers successive residual layers before downsampling occurs.

        # Set of residual layers #1.
        num_layers = 6
        num_filters = 64
        for i in range(num_layers):
            with tf.variable_scope('res1_%d' % (i)) as scope:
                res = self.res_layer(nodes[-1], num_filters)
                nodes.append(res)

        # In-between two sets of residual layers, we must maxpool.
        with tf.variable_scope('pool1') as scope:
            pool_1 = tf.nn.max_pool(nodes[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)
            nodes.append(pool_1)

        # Set of residual layers #2.
        num_layers = 8
        num_filters = 128
        for i in range(num_layers):
            with tf.variable_scope('res2_%d' % (i)) as scope:
                res = self.res_layer(nodes[-1], num_filters)
                nodes.append(res)

        # with tf.variable_scope('pool2') as scope:
        #     pool_2 = tf.nn.max_pool(nodes[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope.name)
        #     nodes.append(pool_2)

        # # Set of residual layers #3.
        # num_layers = 12
        # num_filters = 256
        # for i in range(num_layers):
        #     with tf.variable_scope('res3_%d' % (i)) as scope:
        #         res = self.res_layer(nodes[-1], num_filters)
        #         nodes.append(res)

        # End of residual layers. As per the paper, no maxpooling or dropout occurs at the end of the final convolution.
        dropout = tf.nn.dropout(nodes[-1], 0.5)
        # Global average pooling.
        avg_pool = tf.reduce_mean(dropout, [1, 2])

        # Finally, fully-connected layer which outputs the logits for classification.
        num_classes = 10
        logits = self.dense_layer(avg_pool, num_filters, num_classes)
        
        self.logits = logits


    # A convolutional layer with batch normalization, but without pooling or activation.
    def conv_layer(self, x, filter_height, filter_width, filter_depth, num_filters, stride):
        # Convolution.
        kernel = variable_on_cpu('weights',
                                  shape=[filter_height, filter_width, filter_depth, num_filters],
                                  initializer=tf.contrib.layers.xavier_initializer())
        biases = variable_on_cpu('biases', [num_filters], tf.constant_initializer(0.0))
        conv_first_step = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv_first_step, biases)

        # Batch normalization right after convolution (before any activation).
        # Reference: https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        mean, var = tf.nn.moments(conv, [0])
        offset = tf.Variable(tf.zeros([num_filters]))
        scale = tf.Variable(tf.ones([num_filters]))
        var_epsilon = 1e-3
        bn = tf.nn.batch_normalization(conv, mean, var, offset, scale, var_epsilon)

        return bn

    # A residual layer, as implemented in the ResNet paper.
    def res_layer(self, x, num_filters):
        # Input dimensions.
        x_height = x.get_shape().as_list()[1]
        x_width = x.get_shape().as_list()[2]
        x_depth = x.get_shape().as_list()[3]

        # One residual layer is made up of two successive convolutional layers.
        filter_size = 3
        stride = 1

        # Convolutional layer 1.
        with tf.variable_scope('conv1') as scope:
            conv_1 = self.conv_layer(x, filter_size, filter_size, x_depth, num_filters, stride)
            # ReLU activation.
            activation_1 = tf.nn.relu(conv_1)

        # Convolutional layer 2.
        with tf.variable_scope('conv2') as scope:
            conv_2 = self.conv_layer(activation_1, filter_size, filter_size, num_filters, num_filters, stride)
            # Add initial input through shortcut connection (defined as F + x in the paper) before activation.
            inpt = x
            # Note: When changing residual layers, F and x will have mismatched dimensions. We must increase the dimensionality
            # of x to match F for the addition. We go with Option A from the ResNet paper and simply zero-pad x.
            if x_depth != num_filters:
                # Only the depth will be mismatched, so simply pad in that dimension.
                inpt = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, num_filters - x_depth]], constant_values=0)
            res = tf.add(conv_2, inpt)
            activation_2 = tf.nn.relu(res)

        return activation_2

    # The final fully-connected layer, for which the output is the logits for classification.
    def dense_layer(self, x, x_size, num_classes):
        kernel = variable_on_cpu('weights',
                          shape=[x_size, num_classes],
                          initializer=tf.contrib.layers.xavier_initializer())
        biases = variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))
        dense = tf.matmul(x, kernel) + biases

        return dense


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

import tensorflow as tf

from config import cfg
from utils import get_train_batch, variable_on_cpu

class CNN(object):
    def __init__(self, input_shape, is_training=True, use_test_queue=False):
        self.input_shape = input_shape
        self.name = "cnn"
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_train_batch(cfg.dataset, cfg.batch_size, cfg.num_threads, samples_per_epoch=cfg.samples_per_epoch)
                self.inference(self.X)
                self.loss()
                self._summary()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                if use_test_queue:
                    self.X, self.labels = get_test_batch(cfg.dataset, cfg.batch_size, cfg.num_threads, samples_per_epoch=cfg.samples_per_epoch)
                else:
                    self.X = tf.placeholder(tf.float32, shape=self.input_shape)
                    self.labels = tf.placeholder(tf.int32, shape=(self.input_shape[0],))
                self.inference(self.X, keep_prob=1.0)
                self.loss()
                self.error()


    def inference(self, inputs, keep_prob=0.5):
        with tf.variable_scope('conv1') as scope:
            kernel = variable_on_cpu('weights',
                                      shape=[5, 5, inputs.shape[-1].value, 256],
                                      initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='VALID')
            biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)

        with tf.variable_scope('conv2') as scope:
            kernel = variable_on_cpu('weights',
                                      shape=[5, 5, conv1.shape[-1].value, 256],
                                      initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='VALID')
            biases = variable_on_cpu('biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)

        with tf.variable_scope('conv3') as scope:
            kernel = variable_on_cpu('weights',
                                      shape=[5, 5, conv2.shape[-1].value, 128],
                                      initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(conv2, kernel, [1, 1, 1, 1], padding='VALID')
            biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)

        with tf.variable_scope('fc1') as scope:
            reshape = tf.reshape(conv3, [conv3.shape[0].value, -1])
            weights = variable_on_cpu('weights',
                                       shape=[reshape.shape[1].value, 328],
                                       initializer=tf.contrib.layers.xavier_initializer())
            biases = variable_on_cpu('biases', [328], tf.constant_initializer(0.0))
            fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

        with tf.variable_scope('fc2') as scope:
            weights = variable_on_cpu('weights',
                                       shape=[fc1.shape[1].value, 192],
                                       initializer=tf.contrib.layers.xavier_initializer())
            biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
            fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

        with tf.variable_scope('dropout') as scope:
            dropout = tf.nn.dropout(fc2, keep_prob)
            weights = variable_on_cpu('weights',
                                       shape=[dropout.shape[1].value, 10],
                                       initializer=tf.contrib.layers.xavier_initializer())

            biases = variable_on_cpu('biases', [10], tf.constant_initializer(0.0))
            logits = tf.matmul(dropout, weights) + biases

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

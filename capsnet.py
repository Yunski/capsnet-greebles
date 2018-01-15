import numpy as np
import tensorflow as tf

from utils import get_train_batch, get_test_batch
from config import cfg

class CapsNet(object):
    def __init__(self, input_shape, num_classes, is_training=True, use_test_queue=False):
        self.input_shape = input_shape
        self.name = "capsnet"
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X, self.labels = get_train_batch(cfg.dataset, cfg.batch_size, cfg.num_threads, samples_per_epoch=cfg.samples_per_epoch)
                self.Y = tf.one_hot(self.labels, depth=num_classes, axis=1, dtype=tf.float32)
                self.inference(num_classes)
                self._loss()
                self._summary()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
            else:
                if use_test_queue:
                    self.X, self.labels = get_test_batch(cfg.dataset, cfg.test_batch_size, cfg.num_threads)
                    self.Y = tf.one_hot(self.labels, depth=num_classes, axis=1, dtype=tf.float32)
                else:
                    self.X = tf.placeholder(tf.float32, shape=self.input_shape)
                    self.labels = tf.placeholder(tf.int32, shape=(self.input_shape[0],))
                    self.Y = tf.one_hot(self.labels, depth=num_classes, axis=1, dtype=tf.float32) 
                self.inference(num_classes)
                errors = tf.not_equal(tf.to_int32(self.labels), self.predictions)
                self.error_rate = tf.reduce_mean(tf.cast(errors, tf.float32))          

                self._loss()
                self._summary()


    def inference(self, num_classes, eps=1e-9):
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(n_caps=32, v_len=8, ksize=9, stride=2, use_routing=False, fc=False)
            caps1 = primaryCaps.forward(conv1) 

        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(n_caps=num_classes, v_len=16, use_routing=True, fc=True)
            caps2 = digitCaps.forward(caps1)

        with tf.variable_scope('Mask'):
            self.v_len = tf.sqrt(tf.reduce_sum(tf.square(caps2), axis=2, keep_dims=True) + eps)
            self.v_probs = tf.nn.softmax(self.v_len, dim=1)
            self.predictions = tf.to_int32(tf.argmax(self.v_probs, axis=1))
            self.predictions = tf.reshape(self.predictions, shape=(self.input_shape[0], ))
            masked_v = tf.multiply(tf.squeeze(caps2), tf.reshape(self.Y, (-1, num_classes, 1)))
        
        with tf.variable_scope('Decoder'):
            v_j = tf.reshape(masked_v, shape=(self.input_shape[0], -1))
            fc1 = tf.contrib.layers.fully_connected(v_j, num_outputs=512)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=int(np.prod(self.input_shape[1:3])), activation_fn=tf.sigmoid)


    def _loss(self, m_plus=0.9, m_minus=0.1, down_weighting=0.5, reg=0.0005):
        T = self.Y
        max_plus = tf.square(tf.maximum(0., m_plus - self.v_len))
        max_minus = tf.square(tf.maximum(0., self.v_len - m_minus))
        max_plus = tf.reshape(max_plus, shape=(self.input_shape[0], -1))
        max_minus = tf.reshape(max_minus, shape=(self.input_shape[0], -1))
        L = T * max_plus + down_weighting * (1 - T) * max_minus 
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1))
        X = tf.reshape(self.X, shape=(self.input_shape[0], -1)) 
        square_residual = tf.square(self.decoded - X)
        self.reconstruction_err = tf.reduce_mean(square_residual)        
        self.total_loss = self.margin_loss + reg * np.prod(self.input_shape[1:3]) * self.reconstruction_err


    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=self.input_shape)
        train_summary.append(tf.summary.image('input_img', self.X, max_outputs=5))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img, max_outputs=5))
        self.train_summary = tf.summary.merge(train_summary)

        errors = tf.not_equal(tf.to_int32(self.labels), self.predictions)
        self.error_rate = tf.reduce_mean(tf.cast(errors, tf.float32))


class CapsLayer(object):
    def __init__(self, n_caps, v_len, use_routing=True, fc=True, ksize=None, stride=None, activation_fn=tf.nn.relu, initializer=tf.random_normal_initializer(stddev=0.01)):
        self.n_caps = n_caps
        self.v_len = v_len
        self.use_routing = use_routing
        self.fc = fc
        self.ksize = ksize
        self.stride = stride
        if not self.fc and (not self.ksize or not self.stride):
            raise ValueError("must specify both kernel size and stride for convolutional capslayer.")
        self.activation_fn = activation_fn
        self.initializer = initializer
    

    def forward(self, inputs):
        if self.fc:
            if self.use_routing:
                inputs = tf.reshape(inputs, shape=(inputs.shape[0].value, -1, 1, inputs.shape[-2].value, 1))
                with tf.variable_scope('routing'):
                    b = tf.zeros((inputs.shape[0].value, inputs.shape[1].value, self.n_caps, 1, 1), dtype=tf.float32)
                    caps_out = self.route(inputs, b)
                    output = tf.squeeze(caps_out, axis=1)
                    return output
            else:
                raise ValueError("CapsLayer with fc=True is DigitCaps and must use routing")
        else:
            if not self.use_routing:
                conv_out = tf.contrib.layers.conv2d(inputs, self.n_caps * self.v_len, 
                                                    self.ksize, self.stride, padding='VALID', 
                                                    activation_fn=self.activation_fn)
                conv_out_reshape = tf.reshape(conv_out, (inputs.shape[0].value, -1, self.v_len, 1))
                output = self.squash(conv_out_reshape)
                return output
            else:
                raise ValueError("CapsLayer with fc=False is PrimaryCaps and does not use routing")


    def route(self, inputs, b, num_iter=3):
        W = tf.get_variable('W', shape=(1, inputs.shape[1].value, self.n_caps, inputs.shape[3].value, self.v_len), 
                            dtype=tf.float32, initializer=self.initializer)   
        W_tile = tf.tile(W, [inputs.shape[0].value, 1, 1, 1, 1])
        inputs_tile = tf.tile(inputs, [1, 1, self.n_caps, 1, 1])
        u = tf.matmul(W_tile, inputs_tile, transpose_a=True)
        # technicality: don't want to apply backprop during iterations
        u_stop_grad = tf.stop_gradient(u)
        for r in range(num_iter):
            c = tf.nn.softmax(b, dim=2)
            if r < num_iter - 1:
                s = tf.multiply(c, u_stop_grad)
                s = tf.reduce_sum(s, axis=1, keep_dims=True)
                v = self.squash(s)
                v_tile = tf.tile(v, [1, inputs.shape[1].value, 1, 1, 1])
                agreement = tf.matmul(u_stop_grad, v_tile, transpose_a=True)
                b += agreement
            else:
                s = tf.multiply(c, u)
                s = tf.reduce_sum(s, axis=1, keep_dims=True)
                v = self.squash(s)
        return v
        

    def squash(self, v, eps=1e-9):
        v_l2_2_norm = tf.reduce_sum(tf.square(v), -2, keep_dims=True)
        scale = v_l2_2_norm / (1 + v_l2_2_norm) / tf.sqrt(v_l2_2_norm + eps)
        v_squash = scale * v
        return v_squash


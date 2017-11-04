import numpy as np

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn, bidirectional_dynamic_rnn

class Embedding(object):
    def __init__(self, vocab_size, embedding_dim, embeddings=None, normalize=False, normalize_weights=None, trainable=True):
        self.vocab_size = vocab_size

        if embeddings:
            self.embedding_matrix = tf.constant(embeddings)
        else:
            self.embedding_matrix = tf.Variable(tf.truncated_normal([self.vocab_size, embedding_dim], stddev=0.1))

        if normalize:
            self.embedding_matrix = self._normalized_embeddings(weights=normalize_weights)

    def _normalized_embeddings(self, weights=None):
        if not weights:
            weights = tf.fill([1, self.vocab_size], (1.0 / self.vocab_size))

        mean = tf.reduce_mean(tf.matmul(weights, self.embedding_matrix), 0, keep_dims=True)
        var = tf.reduce_sum(tf.matmul(weights, tf.pow(self.embedding_matrix - mean, 2.)), 0, keep_dims=True)
        stddev = tf.sqrt(1e-6 + var)

        return (self.embedding_matrix - mean) / stddev

    def __call__(self, x):
        batch_embedded = tf.nn.embedding_lookup(self.embedding_matrix, x)

        return batch_embedded

class Normalize(object):
    def __init__(self, weights=None):
        self.weights = weights

    def _normalized_tensor(self, tensor):
        if not self.weights:
            tensor_shape = tensor.get_shape().as_list()
            weights = tf.constant(tf.fill([1, tensor_shape[0]], 1.0 / tensor_shape[0]))
        else:
            weights = self.weights

        mean = tf.reduce_mean(tf.matmul(weights, tensor), 0, keep_dims=True)
        var = tf.reduce_sum(tf.matmul(weights, tf.pow(tensor - mean, 2.)), 0, keep_dims=True)
        stddev = tf.sqrt(1e-6 + var)

        return (tensor - mean) / stddev

    def __call__(self, tensor):
        return self._normalize_tensor(tensor)

class Attention(object):
    def __init__(self, attention_size):
        self.attention_size = attention_size

    def __call__(self, inputs):
        inputs_shape = inputs.get_shape().as_list()

        sequence_length = inputs_shape[1]
        cell_size = inputs_shape[2]

        # Attention mechanism
        W_w = tf.Variable(tf.truncated_normal([cell_size, self.attention_size], stddev=0.1))
        b_w = tf.Variable(tf.truncated_normal([self.attention_size], stddev=0.1))

        u_w = tf.Variable(tf.truncated_normal([self.attention_size], stddev=0.1))

        u_it = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, cell_size]), W_w) + b_w)
        v_u = tf.matmul(u_it, tf.reshape(u_w, [-1, 1]))
        exps = tf.reshape(tf.exp(v_u), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

        # Weigh by alphas
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

        return output, alphas

class BiDirectionalGRU(object):
    def __init__(self, cell_size):
        self.cell_size = cell_size

    def __call__(self, inputs):
        outputs, states = bidirectional_dynamic_rnn(
            GRUCell(self.cell_size),
            GRUCell(self.cell_size),
            inputs=inputs, dtype=tf.float32
        )

        return tf.concat(outputs, 2), states

class Dense(object):
    def __init__(self, hidden_size, n_classes):
        self.hidden_size = hidden_size
        self.n_classes = n_classes

    def __call__(self, inputs):
        in_shape = inputs.get_shape().as_list()

        W_fc1 = tf.Variable(tf.truncated_normal([in_shape[-1], self.hidden_size], stddev=0.1))
        b_fc1 = tf.Variable(tf.truncated_normal([self.hidden_size], stddev=0.1))

        y_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1) + b_fc1)

        W = tf.Variable(tf.zeros([self.hidden_size, self.n_classes]))
        b = tf.Variable(tf.zeros([self.n_classes]))

        return tf.nn.softmax(tf.matmul(y_fc1, W) + b)

import tensorflow as tf
import numpy as np

from keras.datasets import imdb, reuters

import random

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def pad_sequences(sequences, padded_len):
    return np.array([zero_pad(sequence, padded_len) for sequence in sequences])

def zero_pad(arr, padded_len):
    return np.array(arr + ([0] * (padded_len - len(arr))), dtype='f')

def gen_batch(x, y, n_seq):
    indices = random.sample(range(len(x)), n_seq)
    return x[indices], y[indices]

def one_hot(n, dim):
    vec = [0] * dim
    vec[n] = 1

    return np.array(vec, dtype='f')

# Load Data
# (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz")
(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz")

# Pre-process
MAX_LEN = max(len(max(x_train, key=len)), len(max(x_test, key=len)))
N_CLASSES = 46

x_train = pad_sequences(x_train, MAX_LEN)
x_test = pad_sequences(x_test, MAX_LEN)
y_train = np.array([one_hot(y, N_CLASSES) for y in y_train], dtype='f')
y_test  = np.array([one_hot(y, N_CLASSES) for y in y_test], dtype='f')

x = tf.placeholder(tf.float32, [None, MAX_LEN])
W = tf.Variable(tf.zeros([MAX_LEN, N_CLASSES]))
b = tf.Variable(tf.zeros([N_CLASSES]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, N_CLASSES])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
trainer = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_answer = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_answer, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print 'Training...\n'

    for i in range(2000):
        x_batch, y_batch = gen_batch(x_train, y_train, 500)
        # x_batch, y_batch = mnist.train.next_batch(100)

        # print x_batch.shape
        # print y_batch[0]

        if i % 100 == 0:
            print '%d Loss: %f' % (i, sess.run(loss, feed_dict={x: x_batch, y_: y_batch}))
            # print sess.run(y, feed_dict={x: [x_test[0]]})

        sess.run(trainer, feed_dict={x: x_batch, y_: y_batch})

    print
    print 'Loss: ', sess.run(loss, feed_dict={x: x_test, y_: y_test})
    print 'Accuracy: ', sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    # print sess.run(y, feed_dict={x: x_test[:2]})
    # print y_test[:2]

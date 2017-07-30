import tensorflow as tf
import numpy as np

from keras.datasets import imdb

import random

def pad_sequences(sequences, padded_len):
    return np.array([zero_pad(sequence, padded_len) for sequence in sequences])

def zero_pad(arr, padded_len):
    return np.array(arr + ([0] * (padded_len - len(arr))), dtype='f')

def gen_batch(x, y, n_seq):
    indices = random.sample(range(len(x)), n_seq)
    return x[indices], y[indices]

# Load Data
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz")

# Pre-process
MAX_LEN = len(max(x_train + x_test, key=len))

x_train = pad_sequences(x_train, MAX_LEN)
x_test = pad_sequences(x_test, MAX_LEN)
y_train = np.array([[1 & int(not y), 1 & y] for y in y_train], dtype='f')
y_test  = np.array([[1 & int(not y), 1 & y] for y in y_test], dtype='f')

x = tf.placeholder(tf.float32, [None, MAX_LEN])
W = tf.Variable(tf.zeros([MAX_LEN, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
trainer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct_answer = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_answer, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print 'Training...\n'

    for i in range(1000):
        x_batch, y_batch = gen_batch(x_train, y_train, 100)

        if i % 100 == 0:
            print '%d Loss: %f' % (i, sess.run(loss, feed_dict={x: x_batch, y_: y_batch}))

        sess.run(trainer, feed_dict={x: x_batch, y_: y_batch})

    print
    print 'Loss: ', sess.run(loss, feed_dict={x: x_test, y_: y_test})
    print 'Accuracy: ', sess.run(accuracy, feed_dict={x: x_test, y_: y_test})
    print sess.run(y, feed_dict={x: x_test[:2]})
    print y_test[:2]

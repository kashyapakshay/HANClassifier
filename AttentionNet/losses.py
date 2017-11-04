import tensorflow as tf

class CrossEntropyLoss(object):
    def __init__(self):
        pass

    def __call__(self, labels, predicted):
        return tf.reduce_mean(-tf.reduce_sum(labels * tf.log(predicted), reduction_indices=[1]))

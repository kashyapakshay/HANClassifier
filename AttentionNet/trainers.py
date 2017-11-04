import numpy as np
import tensorflow as tf

class BaseTrainer(object):
    def __init__(self, network, loss_instance, optimizer_instance, learning_rate=0.05, batch_size=None, epochs=5):
        self.network = network

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.loss_fn = loss_instance()
        self.optimizer_instance = optimizer_instance

class SimpleTrainer(BaseTrainer):
    def __init__(self, network, loss_instance, optimizer_instance, learning_rate=0.05, batch_size=None, epochs=5):
        BaseTrainer.__init__(self, network, loss_instance, optimizer_instance,
            batch_size=batch_size, epochs=epochs)

    def train(self, training_set, test_set):
        x_train, y_train = training_set[0], training_set[1]
        x_test, y_test = test_set[0], test_set[1]

        inputs = tf.placeholder(tf.int32, [None, x_train.shape[-1]])
        labels = tf.placeholder(tf.float32, [None, self.network.n_classes])

        predicted = self.network(inputs)

        loss = self.loss_fn(labels, predicted)
        optimizer = self.optimizer_instance(self.learning_rate).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            total_size = len(x_train)
            batch_size = self.batch_size or total_size
            n_batches = total_size / batch_size

            for epoch in range(self.epochs):
                print '=== Epoch %d ===' % epoch

                epoch_loss = 0.0

                for i_batch in range(n_batches):
                    start_index = i_batch * self.batch_size
                    end_index = start_index + self.batch_size

                    x_batch, y_batch = x_train[start_index: end_index], y_train[start_index: end_index]

                    batch_loss = sess.run(loss, feed_dict={inputs: x_batch, labels: y_batch})

                    print i_batch, ': ', batch_loss

                    sess.run(optimizer, feed_dict={inputs: x_batch, labels: y_batch})

            correct_answer = tf.equal(tf.argmax(labels, 1), tf.argmax(predicted, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_answer, tf.float32))

            print sess.run(accuracy, feed_dict={inputs: x_test, labels: y_test})

class AdversarialTrainer(BaseTrainer):
    def __init__(self, network, loss_instance, optimizer_instance, learning_rate=0.05, batch_size=None, epochs=5):
        BaseTrainer.__init__(self, network, loss_instance, optimizer_instance,
            batch_size=batch_size, epochs=epochs)

    def _scale_l2(self, x, norm_length):
        x_unit = x / (tf.norm(x, ord=2) + 1e-6)
        return norm_length * x_unit

    def perturb_inputs(self, inputs, loss):
        grad, = tf.gradients(loss, inputs,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)

        perturb = self._scale_l2(grad, 5.0)

        return inputs + perturb

    def train(self, training_set, test_set):
        x_train, y_train = training_set[0], training_set[1]
        x_test, y_test = test_set[0], test_set[1]

        labels = tf.placeholder(tf.float32, [None, self.network.n_classes])

        # Original Inputs
        inputs = tf.placeholder(tf.int32, [None, x_train.shape[-1]])
        predicted = self.network(inputs)

        loss = self.loss_fn(labels, predicted)

        # Perturb the original inputs using loss gradient
        inputs_perturbed = self.perturb_inputs(inputs, loss)
        predicted_perturbed = self.network(inputs_perturbed)

        loss_perturbed = self.loss_fn(labels, predicted_perturbed)

        # Adversarial Loss
        adv_loss = loss

        optimizer = self.optimizer_instance(self.learning_rate).minimize(adv_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            total_size = len(x_train)
            batch_size = self.batch_size or total_size
            n_batches = total_size / batch_size

            for epoch in range(self.epochs):
                print '=== Epoch %d ===' % epoch

                epoch_loss = 0.0

                for i_batch in range(n_batches):
                    start_index = i_batch * self.batch_size
                    end_index = start_index + self.batch_size

                    x_batch, y_batch = x_train[start_index: end_index], y_train[start_index: end_index]

                    batch_loss = sess.run(loss, feed_dict={inputs: x_batch, labels: y_batch})

                    print i_batch, ': ', batch_loss

                    sess.run(optimizer, feed_dict={inputs: x_batch, labels: y_batch})

            correct_answer = tf.equal(tf.argmax(labels, 1), tf.argmax(predicted, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_answer, tf.float32))

            print sess.run(accuracy, feed_dict={inputs: x_test, labels: y_test})

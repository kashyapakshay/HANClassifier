from json import load
import numpy as np

from utils import *

from keras.datasets import imdb, reuters

from layers import *
from losses import CrossEntropyLoss
from trainers import SimpleTrainer

class AttentionNet(object):
    def __init__(self, vocab_size, n_classes):
        self.n_classes = n_classes

        embedding_dim = 300
        gru_cell_size = 100
        attention_size = 50
        hidden_size = 100

        self.embedding_layer = Embedding(vocab_size, embedding_dim)
        self.bi_gru_layer = BiDirectionalGRU(gru_cell_size)
        self.attention_layer = Attention(attention_size)
        self.dense_layer = Dense(hidden_size, self.n_classes)

    def __call__(self, inputs):
        embeddings_in = self.embedding_layer(inputs)
        gru_out, states = self.bi_gru_layer(embeddings_in)
        attention_out, alphas = self.attention_layer(gru_out)
        y = self.dense_layer(attention_out)

        return y

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz")

    n_classes = 46
    MAX_LEN = 20

    vocab_size = get_vocabulary_size(x_train)
    vocab_size = max(vocab_size, get_vocabulary_size(x_test))

    x_train = pad_sequences(x_train, MAX_LEN)
    x_test = pad_sequences(x_test, MAX_LEN)
    y_train = np.array([one_hot(y, n_classes) for y in y_train], dtype='f')
    y_test = np.array([one_hot(y, n_classes) for y in y_test], dtype='f')

    attention_net = AttentionNet(vocab_size, n_classes)

    trainer = SimpleTrainer(attention_net, CrossEntropyLoss, tf.train.GradientDescentOptimizer,
        batch_size=500, epochs=3)
    trainer.train((x_train, y_train), (x_test, y_test))

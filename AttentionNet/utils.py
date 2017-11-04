import random
import numpy as np

def zero_pad(arr, padded_len):
    if padded_len < len(arr):
        return np.array(arr[:padded_len])

    return np.array(arr + ([0] * (padded_len - len(arr))), dtype='f')

def pad_sequences(sequences, padded_len):
    return np.array([zero_pad(sequence, padded_len) for sequence in sequences])

def gen_batch(x, y, n_seq):
    indices = random.sample(range(len(x)), n_seq)
    return x[indices], y[indices]

def one_hot(n, dim, offset=0):
    vec = [0] * dim
    vec[n + offset] = 1

    return np.array(vec, dtype='f')

def get_vocabulary_size(X):
    return max([max(x) for x in X]) + 1

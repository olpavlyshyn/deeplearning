import numpy as np


def generate_minibatches(X, Y, minibatch_size):
    minibatches = []
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))
    num_complete_minibatches = m // minibatch_size

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * minibatch_size: (k + 1) * minibatch_size]
        mini_batch_Y = shuffled_Y[:, k * minibatch_size: (k + 1) * minibatch_size]
        minibatches.append((mini_batch_X, mini_batch_Y))
    if m % minibatch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * minibatch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * minibatch_size:]
        minibatches.append((mini_batch_X, mini_batch_Y))

    return minibatches

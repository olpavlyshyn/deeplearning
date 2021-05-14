import numpy as np


def compute_cost(AL, Y, l2_reg=False, W=None, lambd=None):
    m = Y.shape[1]
    cost = (1 / m) * (-(Y @ np.log(AL).T) - ((1 - Y) @ np.log(1 - AL).T))
    if l2_reg:
        cost = cost + (lambd / (2 * m)) * sum(([np.sum(Wl ** 2) for Wl in W])) # L2 regularization
    return np.squeeze(cost)

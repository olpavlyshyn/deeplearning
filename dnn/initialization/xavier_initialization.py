import numpy as np


def xavier_initialization(units):
    L = len(units)
    W = []
    b = []
    for l in range(1, L):
        Wl = np.random.randn(units[l][0], units[l - 1][0]) * np.sqrt(1. / units[l - 1][0])
        bl = np.zeros((units[l][0], 1))
        W.append(Wl)
        b.append(bl)
    return W, b

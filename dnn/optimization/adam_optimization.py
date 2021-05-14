import numpy as np


def adam_optimization(beta1, beta2, t, W, dW, vdW, sdW, b, db, vdb, sdb, learning_rate, L, epsilon=1e-8):
    for l in range(L - 1):
        # Moving average of the gradients.
        vdW[l] = beta1 * vdW[l] + (1 - beta1) * dW[l]
        vdb[l] = beta1 * vdb[l] + (1 - beta1) * db[l]
        # Moving average of the squared gradients.
        sdW[l] = beta2 * sdW[l] + (1 - beta2) * np.power(dW[l], 2)
        sdb[l] = beta2 * sdb[l] + (1 - beta2) * np.power(db[l], 2)
        # Update parameters.
        W[l] = W[l] - learning_rate * ((vdW[l] / (1 - np.power(beta1, t))) / (np.sqrt(sdW[l] / (1 - np.power(beta2, t))) + epsilon))
        b[l] = b[l] - learning_rate * ((vdb[l] / (1 - np.power(beta1, t))) / (np.sqrt(sdb[l] / (1 - np.power(beta2, t))) + epsilon))
    return W, b, vdW, sdW, vdb, sdb

import numpy as np


def get_accuracy(Yhat, Y):
    accuracy = np.sum((Yhat == Y) / Y.shape[1])
    print("Accuracy: " + str(accuracy))
    return accuracy

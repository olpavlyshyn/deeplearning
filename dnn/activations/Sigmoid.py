import numpy as np
from dnn.activations.IActivation import IActivation


class Sigmoid(IActivation):

    @staticmethod
    def forward(Z):
        G = 1 / (1 + np.exp(-Z))
        return G

    @staticmethod
    def backward(dA, Z):
        s = Sigmoid.forward(Z)
        dZ = dA * s * (1 - s)
        return dZ

import numpy as np
from dnn.activations.IActivation import IActivation


class ReLU(IActivation):

    @staticmethod
    def forward(Z):
        G = np.maximum(0, Z)
        return G

    @staticmethod
    def backward(dA, Z):
        dZ = np.array(dA, copy=True)
        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0
        return dZ

import matplotlib.pyplot as plt
from dnn.initialization.xavier_initialization import xavier_initialization
from dnn.initialization.random_initialization import random_initialization
from dnn.initialization.he_initialization import he_initialization
from dnn.common.compute_cost import compute_cost
from dnn.common.get_accuracy import get_accuracy
from dnn.common.generate_minibatches import generate_minibatches
from dnn.optimization.momentum_optimization import momentum_optimization
from dnn.optimization.gradient_descent_optimization import gradient_descent_optimization
from dnn.optimization.adam_optimization import adam_optimization
from dnn.learning_rate_dacay.exponential_decay import exponential_decay
from dnn.learning_rate_dacay.exponential_decay_with_fixed_interval import exponential_decay_with_fixed_interval
import numpy as np


class DNN:
    def __init__(self):
        self.units = []
        self.L = lambda: len(self.units)
        self.W = []
        self.b = []

        self.vdW = []
        self.vdb = []
        self.sdW = []
        self.sdb = []

    def input_layer(self, n):
        self.units.append((n, None))
        print(f"Added input layer with {n} untis")

    def add_layer(self, units, activation):
        self.units.append((units, activation))
        print(f"Added {self.L() - 1} layer with {units} units and {activation().__class__.__name__} activation")

    def _initialize_parameters(self, init_method):
        methods = {
            "xavier": xavier_initialization,
            "he": he_initialization,
            "random": random_initialization
        }
        np.random.seed(1)
        self.W, self.b = methods[init_method](self.units)
        for l in range(len(self.W)):
            print(f"Initialized parameters for layer: {l + 1} with shape: W: {self.W[l].shape}, b: {self.b[l].shape}")

    def forward_propagation(self, X):
        Z = []
        A = [X]
        for l in range(1, self.L()):
            Zl = self.W[l - 1] @ A[l - 1] + self.b[l - 1]
            Al = self.units[l][1].forward(Zl)
            Z.append(Zl)
            A.append(Al)
        return Z, A

    def back_propagation(self, Y, Z, A, regularization=None, lambd=None):
        dW = []
        db = []
        AL = A[-1]  # output of nn
        # Initializing the back propagation
        dAl = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        for l in reversed(range(1, self.L())):
            dZl = self.units[l][1].backward(dAl, Z[l - 1])
            m = A[l - 1].shape[1]
            dWl = 1. / m * (dZl @ A[l - 1].T)
            if regularization == "L2":
                dWl = dWl + (lambd / m) * self.W[l - 1]
            dbl = 1. / m * np.sum(dZl, axis=1, keepdims=True)
            dAl = self.W[l - 1].T @ dZl
            dW.append(dWl)
            db.append(dbl)
        return dW[::-1], db[::-1]

    def _update_parameters(self, dW, db, learning_rate, optimization=None, beta1=None, beta2=None, t=None):
        if optimization in ("momentum", "adam"):
            if len(self.vdW) == 0 or len(self.vdb) == 0:
                self.vdW = [np.zeros(W.shape) for W in dW]
                self.vdb = [np.zeros(b.shape) for b in db]
                self.sdW = [np.zeros(W.shape) for W in dW]
                self.sdb = [np.zeros(b.shape) for b in db]
            if optimization == "momentum":
                self.W, self.b, self.vdW, self.vdb = momentum_optimization(beta1, self.W, dW, self.vdW, self.b, db,
                                                                           self.vdb, learning_rate, self.L())
            elif optimization == "adam":
                self.W, self.b, self.vdW, self.sdW, self.vdb, self.sdb = adam_optimization(beta1, beta2, t, self.W, dW,
                                                                           self.vdW, self.sdW, self.b, db,
                                                                           self.vdb, self.sdb, learning_rate, self.L())
        else:
            self.W, self.b = gradient_descent_optimization(self.W, dW, self.b, db, learning_rate, self.L())

    def fit(self, X, Y, init_method="xavier", learning_rate=0.0075, num_epochs=1, minibatch_size=0, show_accuracy=True,
            regularization=None, lambd=None, optimization=None, beta1=None, beta2=None,
            decay=None, decay_rate=None, decay_time_interval=None):

        self._initialize_parameters(init_method)
        costs = []
        i = 0
        for epoch in range(num_epochs):
            batches = [(X, Y)] if minibatch_size == 0 else generate_minibatches(X, Y, minibatch_size)
            for batch in batches:
                (X_batch, Y_batch) = batch
                Z, A = self.forward_propagation(X_batch)
                cost = compute_cost(A[-1], Y_batch, l2_reg=True if regularization == "L2" else False, W=self.W, lambd=lambd)
                dW, db = self.back_propagation(Y_batch, Z, A, regularization=regularization, lambd=lambd)
                i = i + 1
                self._update_parameters(dW, db, learning_rate, optimization=optimization, beta1=beta1, beta2=beta2, t=i)
                if i % 500 == 0:
                    costs.append(cost)
                    print(f"Cost: {np.squeeze(cost)} at {i} iteration")
            if decay == "exponential":
                learning_rate = exponential_decay(learning_rate, epoch, decay_rate)

            elif decay == "exponential_with_fixed_interval":
                learning_rate = exponential_decay_with_fixed_interval(learning_rate, epoch, decay_rate,
                                                                      decay_time_interval)
        plt.plot(costs)
        plt.title("Сost per hundreds of interations")
        plt.xlabel("Hundreds of interations")
        plt.ylabel("Cost")

        if show_accuracy:
            Y_hat = self.predict(X)
            get_accuracy(Y_hat, Y)

    def predict(self, X):
        _, A = self.forward_propagation(X)
        Y_hat = A[-1]
        p = np.zeros((1, X.shape[1]))
        # convert probas to 0/1 predictions
        for i in range(0, Y_hat.shape[1]):
            if Y_hat[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        return p

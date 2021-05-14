def gradient_descent_optimization(W, dW, b, db, learning_rate, L):
    for l in range(L - 1):
        W[l] = W[l] - learning_rate * dW[l]
        b[l] = b[l] - learning_rate * db[l]
    return W, b
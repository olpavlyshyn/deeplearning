def momentum_optimization(beta1, W, dW, vdW, b, db, vdb, learning_rate, L):
    for l in range(L - 1):
        vdW[l] = beta1 * vdW[l] + (1 - beta1) * dW[l]
        vdb[l] = beta1 * vdb[l] + (1 - beta1) * db[l]
        W[l] = W[l] - learning_rate * vdW[l]
        b[l] = b[l] - learning_rate * vdb[l]
    return W, b, vdW, vdb

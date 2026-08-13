import numpy as np

def momentum(dW, db, v, t, beta = 0.9, epsilon = 1e-8):
    vd = v

    vd["dW"] = beta * vd["dW"] + (1 - beta) * dW
    vd["db"] = beta * vd["db"] + (1 - beta) * db

    W_update = vd["dW"] / (1 - np.power(beta, t))
    b_update = vd["db"] / (1 - np.power(beta, t))

    return W_update, b_update, vd
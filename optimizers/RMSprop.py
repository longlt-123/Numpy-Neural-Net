import numpy as np

def rmsprop(dW, db, s, t, beta = 0.9, epsilon = 1e-8):
    sd = s

    sd["dW"] = beta * sd["dW"] + (1 - beta) * np.square(dW)
    sd["db"] = beta * sd["db"] + (1 - beta) * np.square(db)

    W_update = dW / (np.sqrt(sd["dW"] / (1 - np.power(beta, t))) + epsilon)
    b_update = db / (np.sqrt(sd["db"] / (1 - np.power(beta, t))) + epsilon)

    return W_update, b_update, sd
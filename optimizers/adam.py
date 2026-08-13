import numpy as np
import math

def adam(dW, db, v, s, t, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    v["dW"] = beta1 * v["dW"] + (1 - beta1) * dW
    v["db"] = beta1 * v["db"] + (1 - beta1) * db

    s["dW"] = beta2 * s["dW"] + (1 - beta2) * np.square(dW)
    s["db"] = beta2 * s["db"] + (1 - beta2) * np.square(db)

    v_corrected = {}                    
    s_corrected = {}

    v_corrected["dW"] = v["dW"] / (1 - math.pow(beta1, t))
    v_corrected["db"] = v["db"] / (1 - math.pow(beta1, t))
    s_corrected["dW"] = s["dW"] / (1 - math.pow(beta2, t))
    s_corrected["db"] = s["db"] / (1 - math.pow(beta2, t))

    W_update = v_corrected["dW"] / (np.sqrt(s_corrected["dW"]) + epsilon)
    b_update = v_corrected["db"] / (np.sqrt(s_corrected["db"]) + epsilon)

    return W_update, b_update, v, s, v_corrected, s_corrected

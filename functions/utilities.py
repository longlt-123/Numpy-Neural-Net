import numpy as np
import math
from activations import *
from output import *

def forward_propagation(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation = "relu"):
    Z = np.dot(A_prev, W) + b
    linear_cache = (A_prev, W, b)
    activation_cache = Z

    if activation == "linear":
        A = linear(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "softmax":
        A = softmax(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def backward_propagation(dA: np.ndarray, cache, activation = "relu"):
    linear_cache, activation_cache = cache

    if activation == "linear":
        dZ = linear(activation_cache, derivative=True) * dA
    elif activation == "relu":
        dZ = relu(activation_cache, derivative=True) * dA
    elif activation == "sigmoid":
        dZ = sigmoid(activation_cache, derivative=True) * dA
    elif activation == "softmax":
        # Lấy ma trận Jacobian 3D, kích thước (m, c, c)
        jacobian = softmax(activation_cache, derivative=True)
        # Nhân chập: dZ[i, k] = tổng_j(dA[i, j] * jacobian[i, j, k])
        dZ = np.einsum('ijk,ij->ik', jacobian, dA, optimize=True)

    A_prev, W, b = linear_cache
    m = A_prev.shape[0]

    dW = np.dot(A_prev.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)

    return dA_prev, dW, db




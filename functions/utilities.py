import numpy as np
import math
from activations import linear, relu, sigmoid
from output import softmax

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

    dW = np.dot(A_prev.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)

    return dA_prev, dW, db

def convert_targets(targets: np.ndarray, to: str = None, threshold = 0.5):
    if to is None:
        if targets.ndim == 1:
            num_classes = len(set(targets))
            converted_targets = np.eye(num_classes)[targets] #One-hot encoding
        else:
            converted_targets = targets
    
    elif to == "categorical":
        num_classes = len(set(targets))
        converted_targets = np.eye(num_classes)[targets]
    
    elif to == "one_hot":
        idx = np.argmax(targets, axis=-1)
        converted_targets = np.zeros(targets.shape)
        converted_targets[np.arange(targets.shape[0]), idx] = 1
    
    elif to == "binary":
        converted_targets = np.where(targets >= threshold, 1, 0)
    
    elif to == "labels":
        if targets.ndim != 1:
            converted_targets = np.argmax(targets, axis=-1)
        else:
            converted_targets = targets
    
    elif to == "probability":
        converted_targets = softmax(targets)
    
    else:
        raise ValueError(f"Unsupported target format: {to}")

    return converted_targets

def initialize_parameters(input_size, output_size, init_type="he"):
    np.random.seed(1)

    if init_type == "zero":
        parameter = np.zeros((input_size, output_size))
    elif init_type == "one":
        parameter = np.ones((input_size, output_size))
    elif init_type == "random":
        parameter = np.random.randn(input_size, output_size) * 0.01
    elif init_type == "he":
        parameter = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    else:
        raise ValueError(f"Unsupported initialize type: {init_type}")

    return parameter

def initialize_optimizer(W, b, optimizer="adam"):
    v = {}
    s = {}

    if optimizer == "gd":
        return {}, {}
    elif optimizer == "rmsprop":
        s["dW"] = np.zeros_like(W)
        s["db"] = np.zeros_like(b)
    elif optimizer == "momentum":
        v["dW"] = np.zeros_like(W)
        v["db"] = np.zeros_like(b)
    elif optimizer == "adam":
        v["dW"] = np.zeros_like(W)
        v["db"] = np.zeros_like(b)
        s["dW"] = np.zeros_like(W)
        s["db"] = np.zeros_like(b)

    return v, s

import numpy as np
import math

def linear(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    if derivative:
        return np.ones(x.shape)
    return x

def relu(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    if derivative:
        return 1. * (x > 0)
    return np.maximum(x, 0)

def sigmoid(x: np.ndarray, derivative: bool = False) -> np.ndarray:
    s = 1 / (1 + np.exp(-x))

    if derivative:
        return s * (1 - s)
    return s

def leaky_relu(x: np.ndarray, derivative: bool = False, alpha: float = 0.01) -> np.ndarray:
    if derivative:
        return np.where(x > 0, 1.0, alpha)
    
    return np.where(x > 0, x, alpha * x)
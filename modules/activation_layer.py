import numpy as np
from functions.activations import linear, relu, sigmoid, leaky_relu
from functions.output import softmax

from modules.base import Layer

class Activation(Layer):
    def __init__(self, activation = "linear"):
        self.activation = activation
        self.activation_cache = None

    def forward(self, A_prev, training=True):
        self.activation_cache = A_prev

        if self.activation == "linear":
            A = linear(A_prev)
        elif self.activation == "relu":
            A = relu(A_prev)
        elif self.activation == "sigmoid":
            A = sigmoid(A_prev)
        elif self.activation == "softmax":
            A = softmax(A_prev)

        return A

    def backward(self, dA):
        if self.activation == "linear":
            dA_prev = linear(self.activation_cache, derivative=True) * dA
        elif self.activation == "relu":
            dA_prev = relu(self.activation_cache, derivative=True) * dA
        elif self.activation == "sigmoid":
            dA_prev = sigmoid(self.activation_cache, derivative=True) * dA
        elif self.activation == "softmax":
            # Lấy ma trận Jacobian 3D, kích thước (m, c, c)
            jacobian = softmax(self.activation_cache, derivative=True)
            # Nhân chập: dA_prev[i, k] = tổng_j(dA[i, j] * jacobian[i, j, k])
            dA_prev = np.einsum('ijk,ij->ik', jacobian, dA, optimize=True)

        return dA_prev
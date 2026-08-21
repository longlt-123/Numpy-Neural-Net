import numpy as np
from functions.output import softmax
from functions.utilities import initialize_parameters, initialize_optimizer
from optimizers.adam import adam
from optimizers.RMSprop import rmsprop
from optimizers.momentum import momentum
from functions.activations import linear, relu, sigmoid, leaky_relu

from modules.base import Layer

class Dense(Layer):
    def __init__(self, number_neurons, init_type = "he", activation = "linear", regularizer = None, lambd = 0.01, freeze = False):
        # Initialize params
        self.W = None
        self.b = None
        self.A_prev = None

        self.number_neurons = number_neurons
        self.init_type = init_type
        self.activation = activation
        self.regularizer = regularizer
        self.lambd = lambd
        self.freeze = freeze
        # Cache for backprop
        self.dA_prev = None
        self.dW = None
        self.db = None

        self.cache = None

        # Optimizer parameters
        self.v = None
        self.s = None
        self.t = None

    def init_params(self, previous_layer_neurons):
        self.W = initialize_parameters(previous_layer_neurons, self.number_neurons, self.init_type)
        self.b = np.zeros((1, self.number_neurons))

    def init_optimizer(self, optimizer):
        self.v, self.s = initialize_optimizer(self.W, self.b, optimizer)
        self.t = 0

    def forward_propagation(self, A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation = "relu"):
        Z = np.matmul(A_prev, W) + b
        linear_cache = (A_prev, W, b)
        activation_cache = Z

        if activation == "linear":
            A = linear(Z)
        elif activation == "relu":
            A = relu(Z)
        elif activation == "leaky_relu":
            A = leaky_relu(Z)
        elif activation == "sigmoid":
            A = sigmoid(Z)
        elif activation == "softmax":
            A = softmax(Z)

        cache = (linear_cache, activation_cache)
        return A, cache


    def backward_propagation(self, dA: np.ndarray, cache, activation = "relu"):
        linear_cache, activation_cache = cache

        if activation == "linear":
            dZ = linear(activation_cache, derivative=True) * dA
        elif activation == "relu":
            dZ = relu(activation_cache, derivative=True) * dA
        elif activation == "leaky_relu":
            dZ = leaky_relu(activation_cache, derivative=True) * dA
        elif activation == "sigmoid":
            dZ = sigmoid(activation_cache, derivative=True) * dA
        elif activation == "softmax":
            # Lấy ma trận Jacobian 3D, kích thước (m, c, c)
            jacobian = softmax(activation_cache, derivative=True)
            # Nhân chập: dZ[i, k] = tổng_j(dA[i, j] * jacobian[i, j, k])
            dZ = np.einsum('...pq,...p->...q', jacobian, dA, optimize=True)

        A_prev, W, b = linear_cache

        dW = np.einsum("...i,...j->ij", A_prev, dZ,optimize=True)
        reduce_axes = tuple(range(dZ.ndim - 1))

        if len(reduce_axes) > 0:
            db = np.sum(dZ, axis=reduce_axes, keepdims=True)
            db = db.reshape(1, dZ.shape[-1])

        else:
            db = dZ.reshape(1, -1)
        dA_prev = np.matmul(dZ, W.T)

        return dA_prev, dW, db
    
    def forward(self, A_prev, training = True):
        A, self.cache = self.forward_propagation(A_prev, self.W, self.b, self.activation)
        return A

    def backward(self, dA):
        self.dA_prev, self.dW, self.db = self.backward_propagation(dA, self.cache, self.activation)

        m = dA.shape[0]
        if self.regularizer == "l1":
            self.dW += (self.lambd / m) * np.sign(self.W)
        elif self.regularizer == "l2":
            self.dW += (self.lambd / m) * self.W

        return self.dA_prev

    def update_parameters(self, learning_rate = 0.01, optimizer=None):
        if self.freeze:
            return

        if optimizer == "rmsprop":
            self.t += 1
            W_update, b_update, self.s = rmsprop(self.dW, self.db, self.s, self.t)
        elif optimizer == "momentum":
            self.t += 1
            W_update, b_update, self.v = momentum(self.dW, self.db, self.v, self.t)
        elif optimizer == "adam":
            self.t += 1
            W_update, b_update, self.v, self.s, _, _ = adam(self.dW, self.db, self.v, self.s, self.t)
        else:
            W_update = self.dW
            b_update = self.db
        
        self.W -= learning_rate * W_update
        self.b -= learning_rate * b_update

    def get_regularization_penalty(self, m):
        if self.regularizer == "l1":
            return (self.lambd / m) * np.sum(np.abs(self.W))
        elif self.regularizer == "l2":
            return (self.lambd / (2 * m)) * np.sum(np.square(self.W))
        return 0.0



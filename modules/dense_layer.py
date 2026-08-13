import numpy as np
from functions.utilities import initialize_parameters, forward_propagation, backward_propagation

class Dense():
    def __init__(self, number_neurons, init_type = "he", activation = "relu", regularizer = None, lambd = 0.01, freeze = False):
        # Initialize params
        self.W = None
        self.b = None
        self.A_prev = None

        self.number_neurons = number_neurons
        self.init_type = init_type
        self.activation = activation
        self.regularizer = regularizer
        self.lambd = lambd
        self.freeze = False

        # Cache for backprop
        self.dA_prev = None
        self.dW = None
        self.db = None

        self.cache = None

    def init_params(self, previous_layer_neurons):
        self.W = initialize_parameters(previous_layer_neurons, self.number_neurons, self.init_type)
        self.b = np.zeros((self.number_neurons , 1))

    def update_parameters(self, learning_rate = 0.01):
        if self.freeze:
            pass
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def forward(self, A_prev):
        A, self.cache = forward_propagation(A_prev, self.W, self.b, self.activation)
        return A, self.cache

    def backward(self, dA):
        self.dA_prev, self.dW, self.db = backward_propagation(dA, self.cache, self.activation)

        m = dA.shape[0]
        if self.regularizer == "l1":
            self.dW += (self.lambd / m) * np.sign(self.W)
        elif self.regularizer == "l2":
            self.dW += (self.lambd / m) * self.W

        return self.dA_prev

    def get_regularization_penalty(self, m):
        if self.regularizer == "l1":
            return (self.lambd / m) * np.sum(np.abs(self.W))
        elif self.regularizer == "l2":
            return (self.lambd / (2 * m)) * np.sum(np.square(self.W))
        return 0.0



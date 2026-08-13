import numpy as np
from functions.utilities import initialize_parameters, forward_propagation, backward_propagation

class Dense():
    def __init__(self, input, number_neurons, init_type="he", activation="relu", regularizer="h2"):
        m = input.shape[0]
        # Initialize params
        self.W = initialize_parameters(m, number_neurons, init_type)
        self.b = np.zeros((number_neurons , 1))
        self.A_prev = input
        self.activation = activation
        self.regularizer = regularizer

        # Cache for backprop
        self.dA_prev = None
        self.dW = None
        self.db = None

        self.cache = None

    def update_parameters(self, learning_rate = 0.01):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def forward(self):
        A, self.cache = forward_propagation(self.A_prev, self.W, self.b, self.activation)
        return A, self.cache

    def backward(self, dA):
        self.dA_prev, self.dW, self.db = backward_propagation(dA, self.cache, self.activation)
        return self.dA_prev



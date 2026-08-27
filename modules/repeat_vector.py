import numpy as np
from modules.base import Layer

class RepeatVector(Layer):
    def __init__(self, n_repeats):
        self.n_repeats = n_repeats
        self.A_prev = None

    def init_params(self, previous_layer_neurons):
        pass
    
    def init_optimizer(self, optimizer):
        pass
    
    def forward(self, A_prev, training=True):
        self.A_prev = A_prev
        return np.repeat(np.expand_dims(A_prev, axis=1), self.n_repeats, axis=1)

    def backward(self, dA):
        return np.sum(dA, axis=1)
    
    def get_regularization_penalty(self, m):
        return 0.0
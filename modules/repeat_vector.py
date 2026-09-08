import numpy as np
from modules.base import Layer

class RepeatVector(Layer):
    def __init__(self, n_repeats, axis):
        self.n_repeats = n_repeats
        self.axis = axis
        self.A_prev = None

    def init_params(self, previous_layer_neurons):
        pass
    
    def init_optimizer(self, optimizer):
        pass
    
    def forward(self, A_prev, training=True):
        self.A_prev = A_prev
        return np.repeat(np.expand_dims(A_prev, axis=self.axis), self.n_repeats, axis=self.axis)

    def backward(self, dA):
        return np.sum(dA, axis=self.axis)
    
    def get_regularization_penalty(self, m):
        return 0.0
import numpy as np

from base import Layer

class Dropout(Layer):
    def __init__(self, keep_prob = 0.8):
        self.keep_prob = keep_prob
        self.mask = None

    def init_params(self, previous_layer_neurons):
        pass

    def update_parameters(self):
        pass

    def forward(self, A_prev, training = True):
        if training:
            random_matrix = np.random.rand(*A_prev.shape)
            self.mask = random_matrix < self.keep_prob
            A = (A_prev * self.mask) / self.keep_prob
        else:
            return A_prev

        return A

    def backward(self, dA):
        dA_prev = (dA * self.mask) / self.keep_prob
        return dA_prev
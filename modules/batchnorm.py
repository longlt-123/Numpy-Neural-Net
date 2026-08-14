import numpy as np
from functions.utilities import initialize_optimizer
from optimizers.adam import adam
from optimizers.RMSprop import rmsprop
from optimizers.momentum import momentum

from modules.base import Layer

class BatchNorm(Layer):
    def __init__(self, momentum=0.9, freeze = False):
        self.freeze = freeze
        self.number_neurons = None
        self.moving_mean = None
        self.moving_var = None
        self.batch_mean = None
        self.batch_var = None

        self.gamma = None
        self.beta = None
        self.momentum = momentum

        self.dA_norm = None
        self.d_gamma = None
        self.d_beta = None
        self.cache = None

        self.v = None
        self.s = None
        self.t = None
    
    def init_params(self, previous_layer_neurons):
        self.gamma = np.ones((1, previous_layer_neurons))
        self.beta = np.zeros((1, previous_layer_neurons))
        self.number_neurons = previous_layer_neurons

        self.moving_mean = np.zeros((1, previous_layer_neurons))
        self.moving_var = np.ones((1, previous_layer_neurons))

    def init_optimizer(self, optimizer):
        self.v, self.s = initialize_optimizer(self.gamma, self.beta, optimizer)
        self.t = 0

    def forward(self, A_prev, training = True):
        if training:
            self.batch_mean = np.mean(A_prev, axis=0)
            self.batch_var = np.var(A_prev, axis=0)
            
            self.A_norm = (A_prev - self.batch_mean) / np.sqrt(self.batch_var + 1e-8)
            
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.batch_mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.batch_var
        else:
            
            self.A_norm = (A_prev - self.moving_mean) / np.sqrt(self.moving_var + 1e-8)

        A = self.gamma * self.A_norm + self.beta 
        return A

    def backward(self, dA):
        m = dA.shape[0]

        self.d_gamma = np.sum(dA * self.A_norm, axis=0, keepdims=True)
        self.d_beta = np.sum(dA, axis=0, keepdims=True)
        
        std_inv = 1.0 / np.sqrt(self.batch_var + 1e-8)
        
        dA_prev = (1.0 / m) * self.gamma * std_inv * (m * dA - self.d_beta - self.A_norm * self.d_gamma)
        
        return dA_prev

    def update_parameters(self, learning_rate = 0.01, optimizer=None):
        if self.freeze:
            return

        if optimizer == "rmsprop":
            self.t += 1
            gamma_update, beta_update, self.s = rmsprop(self.d_gamma, self.d_beta, self.s, self.t)
        elif optimizer == "momentum":
            self.t += 1
            gamma_update, beta_update, self.v = momentum(self.d_gamma, self.d_beta, self.v, self.t)
        elif optimizer == "adam":
            self.t += 1
            gamma_update, beta_update, self.v, self.s, _, _ = adam(self.d_gamma, self.d_beta, self.v, self.s, self.t)
        else:
            gamma_update = self.d_gamma
            beta_update = self.d_beta
        
        self.gamma -= learning_rate * gamma_update
        self.beta -= learning_rate * beta_update
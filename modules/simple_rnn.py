import numpy as np

from functions.activations import relu, linear, sigmoid, leaky_relu
from functions.output import softmax
from functions.utilities import initialize_parameters, forward_propagation, backward_propagation, initialize_optimizer
from optimizers.adam import adam
from optimizers.RMSprop import rmsprop
from optimizers.momentum import momentum

class Simple_RNN():
    def __init__(self, hidden_state_dim, init_type = "he", activation = "softmax", regularizer = None, lambd = 0.01, freeze = False):
        self.x =  None
        self.batch_size = None
        self.n_x = None
        self.T_x = None
        self.T_y = None

        self.n_a = hidden_state_dim

        self.Wax = None
        self.Waa = None
        self.ba = None
        self.Wya = None
        self.by = None

        self.dWax = None
        self.dWaa = None
        self.dba = None
        self.dWya = None
        self.dby = None

        self.a_caches = []
        self.a_prev_caches = []
        self.y_pred_caches = []
        self.xt_caches = self.x

        self.init_type = init_type
        self.activation = activation
        self.regularization = regularizer
        self.lambd = lambd
        self.freeze = freeze

    def one_hot(word, dictionary):
        pass

    def init_params(self, input):
        self.x = input
        self.batch_size = input.shape[0]
        self.n_x = input.shape[2]
        self.T_x = input.shape[1]
        self.T_y = self.T_x
        
        self.Waa = initialize_parameters(1, self.n_a, self.init_type)
        self.Wax = initialize_parameters(self.n_x, self.n_a, self.init_type)
        self.ba = initialize_parameters(1, self.n_a, self.init_type)

        self.Wya = initialize_parameters(self.n_a, self.n_x, self.init_type)
        self.by = initialize_parameters(1, self.n_x, "zero")

        self.dWaa = np.zeros_like(self.Waa)
        self.dWax = np.zeros_like(self.Wax)
        self.dba = np.zeros_like(self.ba)
        self.dWya = np.zeros_like(self.Wya)
        self.dby = np.zeros_like(self.by)

    def rnn_cell_forward(self, xt, a_prev):
        a_next = np.tanh(np.dot(xt, self.Wax) + np.dot(a_prev, self.Waa) + self.ba)
        yt_pred = softmax(np.dot(a_next, self.Wya) + self.by, axis=-1)

        self.a_caches.append(a_next)
        self.a_prev_caches.append(a_prev)
        self.xt_caches.append(xt)
        self.y_pred_caches.append(yt_pred)

        return a_next, yt_pred

    def rnn_forward(self):
        self.a_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.y_pred_caches = np.zeros((self.batch_size, self.T_y, self.n_x))

        a_next = np.zeros((1, self.n_a))
        for t in range(self.T_x):
            xt = self.xt_caches[:,t,:]
            a_next, yt_pred = self.rnn_cell_forward(xt, a_next)
            self.a_caches.append(a_next)
            self.y_pred_caches.append(yt_pred)

        return self.a_caches, self.y_pred_caches, self.x
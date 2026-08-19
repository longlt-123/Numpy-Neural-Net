import numpy as np

from functions.activations import relu, linear, sigmoid, leaky_relu
from functions.output import softmax
from functions.utilities import initialize_parameters, forward_propagation, backward_propagation, initialize_optimizer
from optimizers.adam import adam
from optimizers.RMSprop import rmsprop
from optimizers.momentum import momentum

from base import Layer

class LSTM():
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

        self.Wf = None
        self.bf = None
        self.Wi = None
        self.bi = None
        self.Wc = None
        self.bc = None
        self.Wo = None
        self.bo = None

        self.dWax = None
        self.dWaa = None
        self.dba = None
        self.dWya = None
        self.dby = None

        self.dWf = None
        self.dbf = None
        self.dWi = None
        self.dbi = None
        self.dWc = None
        self.dbc = None
        self.dWo = None
        self.dbo = None

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

        self.Wf = initialize_parameters(self.n_a + self.n_x, self.n_a, "zero")
        self.Wi = initialize_parameters(self.n_a + self.n_x, self.n_a, "zero")
        self.Wc = initialize_parameters(self.n_a + self.n_x, self.n_a, "zero")
        self.Wo = initialize_parameters(self.n_a + self.n_x, self.n_a, "zero")
        self.bf = initialize_parameters(1, self.n_a, "zero")
        self.bi = initialize_parameters(1, self.n_a, "zero")
        self.bc = initialize_parameters(1, self.n_a, "zero")
        self.bo = initialize_parameters(1, self.n_a, "zero")

        self.dWaa = np.zeros_like(self.Waa)
        self.dWax = np.zeros_like(self.Wax)
        self.dba = np.zeros_like(self.ba)
        self.dWya = np.zeros_like(self.Wya)
        self.dby = np.zeros_like(self.by)

        self.dWf = np.zeros_like(self.Wf)
        self.dbf = np.zeros_like(self.bf)
        self.dWi = np.zeros_like(self.Wi)
        self.dbi = np.zeros_like(self.bi)
        self.dWc = np.zeros_like(self.Wc)
        self.dbc = np.zeros_like(self.bc)
        self.dWo = np.zeros_like(self.Wo)
        self.dbo = np.zeros_like(self.bo)

    def lstm_cell_forward(self, xt, a_prev):
        pass

    def lstm_forward(self):
        pass







        

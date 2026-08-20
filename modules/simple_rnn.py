import numpy as np

from functions.activations import relu, linear, sigmoid, leaky_relu
from functions.output import softmax
from functions.utilities import initialize_parameters, forward_propagation, backward_propagation, initialize_optimizer
from optimizers.adam import adam
from optimizers.RMSprop import rmsprop
from optimizers.momentum import momentum

class Simple_RNN():
    def __init__(self, hidden_state_dim, init_type = "he", bidirectional = False, merge_mode = "concat", regularizer = None, lambd = 0.01, freeze = False):
        self.x =  None
        self.batch_size = None
        self.n_x = None
        self.T_x = None
        self.T_y = None

        self.n_a = hidden_state_dim

        self.Wax_right = None
        self.Waa_right = None
        self.ba_right = None
        self.Wya_right = None
        self.by = None

        self.Wax_opp = None
        self.Waa_opp = None
        self.ba_opp = None
        self.Wya_opp = None

        self.dWax_right = None
        self.dWaa_right = None
        self.dba_right = None
        self.dWya_right = None
        self.dby = None

        self.dWax_opp = None
        self.dWaa_opp = None
        self.dba_opp = None
        self.dWya_opp = None

        self.a_right_caches = []
        self.a_prev_right_caches = []
        self.a_opp_caches = []
        self.a_prev_opp_caches = []
        self.y_pred_caches = []
        self.xt_caches = []
        self.dx = []

        self.bidirectional = bidirectional
        self.init_type = init_type
        self.merge_mode = merge_mode
        self.regularization = regularizer
        self.lambd = lambd
        self.freeze = freeze

    def init_optimizer(self, optimizer):
        self.Wa_right = np.concatenate((self.Wax_right, self.Waa_right), axis=0)
        self.ba_right = self.ba_right
        self.v_right, self.s_right = initialize_optimizer(self.Wa_right, self.ba_right, optimizer)

        self.Wa_opp = np.concatenate((self.Wax_opp, self.Waa_opp), axis=0)
        self.ba_opp = self.ba_opp
        self.v_opp, self.s_opp = initialize_optimizer(self.Wa_opp, self.ba_opp, optimizer)
        self.t = 0

    def init_params(self, input_dims):
        self.batch_size = input_dims[0]
        self.n_x = input_dims[2]
        self.T_x = input_dims[1]
        self.T_y = self.T_x
        
        self.Waa_right = initialize_parameters(self.n_a, self.n_a, self.init_type)
        self.Wax_right = initialize_parameters(self.n_x, self.n_a, self.init_type)
        self.ba_right = initialize_parameters(1, self.n_a, self.init_type)

        self.Wya_right = initialize_parameters(self.n_a, self.n_x, self.init_type)
        self.by = initialize_parameters(1, self.n_x, "zero")

        if self.bidirectional:
            self.Waa_opp = initialize_parameters(self.n_a, self.n_a, self.init_type)
            self.Wax_opp = initialize_parameters(self.n_x, self.n_a, self.init_type)
            self.ba_opp = initialize_parameters(1, self.n_a, self.init_type)

            self.Wya_opp = initialize_parameters(self.n_a, self.n_x, self.init_type)

    def rnn_cell_forward(self, xt, a_prev, bidirectional = False):
        if bidirectional == False:
            a_next = np.tanh(np.dot(xt, self.Wax_right) + np.dot(a_prev, self.Waa_right) + self.ba_right)
        else:
            a_next = np.tanh(np.dot(xt, self.Wax_opp) + np.dot(a_prev, self.Waa_opp) + self.ba_opp)
        return a_next

    def forward(self, input, training = True):
        self.x = input

        self.a_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.a_prev_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.xt_caches = np.zeros((self.batch_size, self.T_x, self.n_x))
        self.y_pred_caches = np.zeros((self.batch_size, self.T_y, self.n_x))

        a_right = np.zeros((self.batch_size, self.n_a))
        for t in range(self.T_x):
            xt = self.x[:,t,:]
            self.xt_caches[:,t,:] = xt
            self.a_prev_right_caches[:,t,:] = a_right

            a_right = self.rnn_cell_forward(xt, a_right, bidirectional=False)
            self.a_right_caches[:,t,:] = a_right
            self.y_pred_caches[:,t,:] += np.dot(a_right, self.Wya_right)

        if self.bidirectional:
            self.a_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))

            a_opp = np.zeros((self.batch_size, self.n_a))
            for t in reversed(range(self.T_x)):
                xt = self.x[:,t,:]
                self.a_prev_opp_caches[:,t,:] = a_opp

                a_opp = self.rnn_cell_forward(xt, a_opp, bidirectional=True)
                self.y_pred_caches[:,t,:] += np.dot(a_opp, self.Wya_opp)
                self.a_opp_caches[:,t,:] = a_opp

        self.y_pred_caches = softmax(self.y_pred_caches + self.by, axis=-1)

        return self.a_right_caches, self.a_opp_caches, self.y_pred_caches, self.x

    def rnn_cell_backward(self, dA, xt, a_prev, bidirectional = False):
        if bidirectional == False:
            dtanh = dA * (1 - np.tanh(np.dot(xt, self.Wax_right) + np.dot(a_prev, self.Waa_right) + self.ba_right) ** 2)

            dxt = np.dot(dtanh, self.Wax_right.T)
            da_prev = np.dot(dtanh, self.Waa_right.T)
            dWax_right = np.dot(xt.T, dtanh)
            dWaa_right = np.dot(a_prev.T, dtanh)
            dba_right = np.sum(dtanh, axis=-1, keepdims=True)
            return dxt, da_prev, dWax_right, dWaa_right, dba_right
        else:
            dtanh = dA * (1 - np.tanh(np.dot(xt, self.Wax_opp) + np.dot(a_prev, self.Waa_opp) + self.ba_opp) ** 2)

            dxt = np.dot(dtanh, self.Wax_opp.T)
            da_prev = np.dot(dtanh, self.Waa_opp.T)
            dWax_opp = np.dot(xt.T, dtanh)
            dWaa_opp = np.dot(a_prev.T, dtanh)
            dba_opp = np.sum(dtanh, axis=-1, keepdims=True)
            return dxt, da_prev, dWax_opp, dWaa_opp, dba_opp

    def backward(self, dA, dL = None):
        self.dx = np.zeros_like(self.x)

        self.dWaa_right = np.zeros_like(self.Waa_right)
        self.dWax_right = np.zeros_like(self.Wax_right)
        self.dba_right = np.zeros_like(self.ba_right)
        self.dWya_right = np.zeros_like(self.Wya_right)
        self.dby = np.zeros_like(self.by)

        if self.bidirectional == False:
            dA_right = dA
            dA_opp = None
        else:
            if self.merge_mode == "concat":
                dA_right = dA[:,:,:self.n_a]
                dA_opp = dA[:,:,self.n_a:]
            elif self.merge_mode == "sum":
                dA_right = dA
                dA_opp = dA
            elif self.merge_mode == "average":
                dA_right = dA / 2
                dA_opp = dA / 2
            elif self.merge_mode == "multiply":
                dA_right = dA * self.a_opp_caches
                dA_opp = dA * self.a_right_caches
            
            self.dWaa_opp = np.zeros_like(self.Waa_opp)
            self.dWax_opp = np.zeros_like(self.Wax_opp)
            self.dba_opp = np.zeros_like(self.ba_opp)
            self.dWya_opp = np.zeros_like(self.Wya_opp)

        da_prevt_right = np.zeros((self.batch_size, self.n_a))
        da_prevt_opp = np.zeros((self.batch_size, self.n_a))

        for t in reversed(range(self.T_x)):
            xt = self.xt_caches[:,t,:]
            a_prev_right = self.a_prev_right_caches[:,t,:]
            dA_t_right = dA_right[:,t,:]

            dxt_right, da_prevt_right, dWax_right, dWaa_right, dba_right = self.rnn_cell_backward(da_prevt_right + dA_t_right, xt, a_prev_right, bidirectional=False)

            self.dWax_right += dWax_right
            self.dWaa_right += dWaa_right
            self.dba_right += dba_right
            self.dxt[:,t,:] += dxt_right

            if self.bidirectional:
                a_prev_opp = self.a_prev_opp_caches[:,t,:]
                dA_t_opp = dA_opp[:,t,:]

                dxt_opp, da_prevt_opp, dWax_opp, dWaa_opp, dba_opp = self.rnn_cell_backward(da_prevt_opp + dA_t_opp, xt, a_prev_opp, bidirectional=True)

                self.dWax_opp += dWax_opp
                self.dWaa_opp += dWaa_opp
                self.dba_opp += dba_opp
                self.dxt[:,t,:] += dxt_opp

        m = self.batch_size
        
        if self.regularizer == "l1":
            self.dWaa_right += (self.lambd / m) * np.sign(self.Waa_right)
            self.dWax_right += (self.lambd / m) * np.sign(self.Wax_right)
            self.dba_right += (self.lambd / m) * np.sign(self.ba_right)
            if self.bidirectional:
                self.dWaa_opp += (self.lambd / m) * np.sign(self.Waa_opp)
                self.dWax_opp += (self.lambd / m) * np.sign(self.Wax_opp)
                self.dba_opp += (self.lambd / m) * np.sign(self.ba_opp)
        elif self.regularizer == "l2":
            self.dWaa_right += (self.lambd / m) * self.Waa_right
            self.dWax_right += (self.lambd / m) * self.Wax_right
            self.dba_right += (self.lambd / m) * self.ba_right
            if self.bidirectional:
                self.dWaa_opp += (self.lambd / m) * self.Waa_opp
                self.dWax_opp += (self.lambd / m) * self.Wax_opp
                self.dba_opp += (self.lambd / m) * self.ba_opp
        gradients = {
            "dx": self.dx,
            "dWaa": self.dWaa_right,
            "dWax": self.dWax_right,
            "dba": self.dba_right,
            "dWya": self.dWya_right,
            "dby": self.dby,
        }
        return gradients


    def compute_hidden_state_for_next_layer(self):
        if self.bidirectional:
            if self.merge_mode == "concat":
                return np.concatenate((self.a_right_caches, self.a_opp_caches), axis=-1)
            elif self.merge_mode == "sum":
                return self.a_right_caches + self.a_opp_caches
            elif self.merge_mode == "average":
                return (self.a_right_caches + self.a_opp_caches) / 2
            elif self.merge_mode == "multiply":
                return self.a_right_caches * self.a_opp_caches
        else:
            return self.a_right_caches

    def update_parameters(self, learning_rate = 0.01, optimizer=None):
        if self.freeze:
            return

        if optimizer == "rmsprop":
            self.t += 1
            Wa_right_update, ba_right_update, self.s_right = rmsprop(self.dWax_right, self.dba_right, self.s_right, self.t)
            Wa_opp_update, ba_opp_update, self.s_opp = rmsprop(self.dWax_opp, self.dba_opp, self.s_opp, self.t)
        elif optimizer == "momentum":
            self.t += 1
            Wa_right_update, ba_right_update, self.v_right = momentum(self.dWax_right, self.dba_right, self.v_right, self.t)
            Wa_opp_update, ba_opp_update, self.v_opp = momentum(self.dWax_opp, self.dba_opp, self.v_opp, self.t)
        elif optimizer == "adam":
            self.t += 1
            Wa_right_update, ba_right_update, self.v_right, self.s_right, _, _ = adam(self.dWax_right, self.dba_right, self.v_right, self.s_right, self.t)
            Wa_opp_update, ba_opp_update, self.v_opp, self.s_opp, _, _ = adam(self.dWax_opp, self.dba_opp, self.v_opp, self.s_opp, self.t)
        else:
            Wa_right_update = self.dWax_right
            ba_right_update = self.dba_right
            Wa_opp_update = self.dWax_opp
            ba_opp_update = self.dba_opp

        self.Wa_right -= learning_rate * Wa_right_update
        self.ba_right -= learning_rate * ba_right_update
        self.Wa_opp -= learning_rate * Wa_opp_update
        self.ba_opp -= learning_rate * ba_opp_update

    def get_regularization_penalty(self, m):
        if self.regularizer == "l1":
            return (self.lambd / m) * np.sum(np.abs(self.Wa_right)) + (self.lambd / m) * np.sum(np.abs(self.Wa_opp))
        elif self.regularizer == "l2":
            return (self.lambd / (2 * m)) * np.sum(np.square(self.Wa_right)) + (self.lambd / (2 * m)) * np.sum(np.square(self.Wa_opp))
        return 0.0
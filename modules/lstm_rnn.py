import numpy as np

from functions.activations import relu, linear, sigmoid, leaky_relu
from functions.output import softmax
from functions.utilities import initialize_parameters, initialize_optimizer
from optimizers.adam import adam
from optimizers.RMSprop import rmsprop
from optimizers.momentum import momentum

from modules.base import Layer

class LSTM(Layer):
    def __init__(self, hidden_state_dim, init_type = "he", bidirectional = False, merge_mode = "concat", regularizer = None, lambd = 0.01, freeze = False, return_sequences=True):
        self.x =  None
        self.batch_size = None
        self.n_x = None
        self.T_x = None
        self.T_y = None

        self.n_a = hidden_state_dim

        self.Wf_right = None
        self.Wf_opp = None
        self.bf_right = None
        self.bf_opp = None
        self.Wi_right = None
        self.Wi_opp = None
        self.bi_right = None
        self.bi_opp = None
        self.Wc_right = None
        self.Wc_opp = None
        self.bc_right = None
        self.bc_opp = None
        self.Wo_right = None
        self.Wo_opp = None
        self.bo_right = None
        self.bo_opp = None

        self.dWf_right = None
        self.dWf_opp = None
        self.dbf_right = None
        self.dbf_opp = None
        self.dWi_right = None
        self.dWi_opp = None
        self.dbi_right = None
        self.dbi_opp = None
        self.dWc_right = None
        self.dWc_opp = None
        self.dbc_right = None
        self.dbc_opp = None
        self.dWo_right = None
        self.dWo_opp = None
        self.dbo_right = None
        self.dbo_opp = None

        self.a_state = None
        self.c_state = None
        self.a_state_opp = None
        self.c_state_opp = None

        self.da_0 = None 
        self.dc_0 = None
        self.da_0_opp = None
        self.dc_0_opp = None

        self.a_right_caches = []
        self.a_prev_right_caches = []
        self.a_opp_caches = []
        self.a_prev_opp_caches = []

        self.c_right_caches = []
        self.c_prev_right_caches = []
        self.c_opp_caches = []
        self.c_prev_opp_caches = []

        self.c_hat_right_caches = []
        self.c_hat_opp_caches = []

        self.f_right_caches = []
        self.f_opp_caches = []

        self.i_right_caches = []
        self.i_opp_caches = []

        self.o_right_caches = []
        self.o_opp_caches = []

        self.xt_caches = []
        self.dxt_caches = []

        self.bidirectional = bidirectional
        self.init_type = init_type
        self.merge_mode = merge_mode
        self.regularizer = regularizer
        self.lambd = lambd
        self.freeze = freeze
        self.return_sequences = return_sequences

        self.Vf_right = None
        self.Vf_opp = None
        self.Sf_right = None
        self.Sf_opp = None

        self.Vi_right = None
        self.Vi_opp = None
        self.Si_right = None
        self.Si_opp = None

        self.Vc_right = None
        self.Vc_opp = None
        self.Sc_right = None
        self.Sc_opp = None

        self.Vo_right = None
        self.Vo_opp = None
        self.So_right = None
        self.So_opp = None

        self.t = None

    def init_optimizer(self, optimizer):
        self.Vf_right, self.Sf_right = initialize_optimizer(self.Wf_right, self.bf_right, optimizer)
        self.Vi_right, self.Si_right = initialize_optimizer(self.Wi_right, self.bi_right, optimizer)
        self.Vc_right, self.Sc_right = initialize_optimizer(self.Wc_right, self.bc_right, optimizer)
        self.Vo_right, self.So_right = initialize_optimizer(self.Wo_right, self.bo_right, optimizer)

        if self.bidirectional:
            self.Vf_opp, self.Sf_opp = initialize_optimizer(self.Wf_opp, self.bf_opp, optimizer)
            self.Vi_opp, self.Si_opp = initialize_optimizer(self.Wi_opp, self.bi_opp, optimizer)
            self.Vc_opp, self.Sc_opp = initialize_optimizer(self.Wc_opp, self.bc_opp, optimizer)
            self.Vo_opp, self.So_opp = initialize_optimizer(self.Wo_opp, self.bo_opp, optimizer)

        self.t = 0

    def init_params(self, input_dims):
        self.n_x = input_dims

        self.Wf_right = initialize_parameters(self.n_a + self.n_x, self.n_a, self.init_type)
        self.Wi_right = initialize_parameters(self.n_a + self.n_x, self.n_a, self.init_type)
        self.Wc_right = initialize_parameters(self.n_a + self.n_x, self.n_a, self.init_type)
        self.Wo_right = initialize_parameters(self.n_a + self.n_x, self.n_a, self.init_type)
        self.bf_right = np.ones((1, self.n_a))
        self.bi_right = initialize_parameters(1, self.n_a, self.init_type)
        self.bc_right = initialize_parameters(1, self.n_a, self.init_type)
        self.bo_right = initialize_parameters(1, self.n_a, self.init_type)

        if self.bidirectional:
            self.Wf_opp = initialize_parameters(self.n_a + self.n_x, self.n_a, self.init_type)
            self.Wi_opp = initialize_parameters(self.n_a + self.n_x, self.n_a, self.init_type)
            self.Wc_opp = initialize_parameters(self.n_a + self.n_x, self.n_a, self.init_type)
            self.Wo_opp = initialize_parameters(self.n_a + self.n_x, self.n_a, self.init_type)
            self.bf_opp = np.ones((1, self.n_a))
            self.bi_opp = initialize_parameters(1, self.n_a, self.init_type)
            self.bc_opp = initialize_parameters(1, self.n_a, self.init_type)
            self.bo_opp = initialize_parameters(1, self.n_a, self.init_type)

    def set_states(self, a_state, c_state, a_state_opp=None, c_state_opp=None):
        self.a_state = a_state
        self.c_state = c_state
        self.a_state_opp = a_state_opp
        self.c_state_opp = c_state_opp

    def get_initial_state_gradients(self):
        return self.da_0, self.dc_0, self.da_0_opp, self.dc_0_opp

    def get_last_states(self):
        if self.bidirectional:
            if self.merge_mode == "concat":
                a_state = np.concatenate((self.a_right_caches[:, -1, :], self.a_opp_caches[:, 0, :]), axis=-1)
                c_state = np.concatenate((self.c_right_caches[:, -1, :], self.c_opp_caches[:, 0, :]), axis=-1)
                return a_state, c_state
            elif self.merge_mode == "sum":
                return self.a_right_caches[:, -1, :] + self.a_opp_caches[:, 0, :], self.c_right_caches[:, -1, :] + self.c_opp_caches[:, 0, :]
            elif self.merge_mode == "average":
                return (self.a_right_caches[:, -1, :] + self.a_opp_caches[:, 0, :]) / 2, (self.c_right_caches[:, -1, :] + self.c_opp_caches[:, 0, :]) / 2
            elif self.merge_mode == "multiply":
                return self.a_right_caches[:, -1, :] * self.a_opp_caches[:, 0, :], self.c_right_caches[:, -1, :] * self.c_opp_caches[:, 0, :]
        return self.a_right_caches[:, -1, :], self.c_right_caches[:, -1, :]

    def pass_states_to(self, target_lstm):
        a_state, c_state = self.get_last_states()
        target_lstm.set_states(a_state, c_state)

    def reset_states(self):
        self.a_state = None
        self.c_state = None
        self.a_state_opp = None
        self.c_state_opp = None

    def lstm_cell_forward(self, xt, a_prev, c_prev, bidirectional = False):
        if bidirectional == False:
            f_gate = sigmoid(np.matmul(np.concatenate((a_prev, xt), axis=-1), self.Wf_right) + self.bf_right)
            i_gate = sigmoid(np.matmul(np.concatenate((a_prev, xt), axis=-1), self.Wi_right) + self.bi_right)
            c_hat = np.tanh(np.matmul(np.concatenate((a_prev, xt), axis=-1), self.Wc_right) + self.bc_right)
            o_gate = sigmoid(np.matmul(np.concatenate((a_prev, xt), axis=-1), self.Wo_right) + self.bo_right)
        else:
            f_gate = sigmoid(np.matmul(np.concatenate((a_prev, xt), axis=-1), self.Wf_opp) + self.bf_opp)
            i_gate = sigmoid(np.matmul(np.concatenate((a_prev, xt), axis=-1), self.Wi_opp) + self.bi_opp)
            c_hat = np.tanh(np.matmul(np.concatenate((a_prev, xt), axis=-1), self.Wc_opp) + self.bc_opp)
            o_gate = sigmoid(np.matmul(np.concatenate((a_prev, xt), axis=-1), self.Wo_opp) + self.bo_opp)

        c_next = f_gate * c_prev + i_gate * c_hat
        a_next = o_gate * np.tanh(c_next)

        return a_next, c_next, c_hat, o_gate, i_gate, f_gate
    
    def forward(self, input, training = True):
        self.x = input
        self.batch_size = input.shape[0]
        self.n_x = input.shape[2]
        self.T_x = input.shape[1]
        self.T_y = self.T_x

        self.xt_caches = np.zeros((self.batch_size, self.T_x, self.n_x))

        self.a_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.a_prev_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.a_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.a_prev_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))

        self.c_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.c_prev_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.c_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.c_prev_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))

        self.f_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.f_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))

        self.i_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.i_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))

        self.o_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.o_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))

        self.c_hat_right_caches = np.zeros((self.batch_size, self.T_x, self.n_a))
        self.c_hat_opp_caches = np.zeros((self.batch_size, self.T_x, self.n_a))

        if self.a_state is not None and self.a_state.shape[0] == self.batch_size:
            a_right = self.a_state
        else:
            a_right = np.zeros((self.batch_size, self.n_a))
        
        if self.c_state is not None and self.c_state.shape[0] == self.batch_size:
            c_right = self.c_state
        else:
            c_right = np.zeros((self.batch_size, self.n_a))
        
        for t in range(self.T_x):
            xt = self.x[:,t,:]
            self.xt_caches[:,t,:] = xt
            self.a_prev_right_caches[:,t,:] = a_right
            self.c_prev_right_caches[:,t,:] = c_right

            a_right, c_right, c_hat_right, o_gate_right, i_gate_right, f_gate_right = self.lstm_cell_forward(xt, a_right, c_right, bidirectional=False)

            self.a_right_caches[:,t,:] = a_right
            self.c_right_caches[:,t,:] = c_right
            self.f_right_caches[:,t,:] = f_gate_right
            self.i_right_caches[:,t,:] = i_gate_right
            self.o_right_caches[:,t,:] = o_gate_right
            self.c_hat_right_caches[:,t,:] = c_hat_right
        if self.bidirectional:
            if self.a_state_opp is not None and self.a_state_opp.shape[0] == self.batch_size:
                a_opp = self.a_state_opp
            else:
                a_opp = np.zeros((self.batch_size, self.n_a))
            
            if self.c_state_opp is not None and self.c_state_opp.shape[0] == self.batch_size:
                c_opp = self.c_state_opp
            else:
                c_opp = np.zeros((self.batch_size, self.n_a))

            for t in reversed(range(self.T_x)):
                xt = self.x[:,t,:]
                self.a_prev_opp_caches[:,t,:] = a_opp
                self.c_prev_opp_caches[:,t,:] = c_opp

                a_opp, c_opp, c_hat_opp, o_gate_opp, i_gate_opp, f_gate_opp = self.lstm_cell_forward(xt, a_opp, c_opp, bidirectional=True)

                self.a_opp_caches[:,t,:] = a_opp
                self.c_opp_caches[:,t,:] = c_opp
                self.f_opp_caches[:,t,:] = f_gate_opp
                self.i_opp_caches[:,t,:] = i_gate_opp
                self.o_opp_caches[:,t,:] = o_gate_opp
                self.c_hat_opp_caches[:,t,:] = c_hat_opp
        if training == False:
            self.a_state = a_right
            self.c_state = c_right
            if self.bidirectional:
                self.a_state_opp = a_opp
                self.c_state_opp = c_opp

        A = self.compute_hidden_state_for_next_layer()
        return A

    def compute_hidden_state_for_next_layer(self):
        if self.return_sequences:
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
        else:
            if self.bidirectional:
                if self.merge_mode == "concat":
                    return np.concatenate((self.a_right_caches[:, -1, :], self.a_opp_caches[:, 0, :]), axis=-1)
                elif self.merge_mode == "sum":
                    return self.a_right_caches[:, -1, :] + self.a_opp_caches[:, 0, :]
                elif self.merge_mode == "average":
                    return (self.a_right_caches[:, -1, :] + self.a_opp_caches[:, 0, :]) / 2
                elif self.merge_mode == "multiply":
                    return self.a_right_caches[:, -1, :] * self.a_opp_caches[:, 0, :]
            else:
                return self.a_right_caches[:, -1, :]

    def lstm_cell_backward(self, dA, xt, a_prev, dC, c_next, c_hat, c_prev, i_gate, f_gate, o_gate, bidirectional = False):
        do = dA * np.tanh(c_next) * o_gate * (1 - o_gate)
        dpc = (i_gate * dC + o_gate * (1 - np.tanh(c_next)**2) * i_gate * dA) * (1 - c_hat**2)
        di = (c_hat * dC + o_gate * (1 - np.tanh(c_next)**2) * c_hat * dA) * i_gate * (1 - i_gate)
        df = (c_prev * dC + o_gate * (1 - np.tanh(c_next)**2) * c_prev * dA) * f_gate * (1 - f_gate)
        if bidirectional == False:
            dWf_right = np.matmul(np.concatenate((a_prev, xt), axis=-1).T, df)
            dWc_right = np.matmul(np.concatenate((a_prev, xt), axis=-1).T, dpc)
            dWi_right = np.matmul(np.concatenate((a_prev, xt), axis=-1).T, di)
            dWo_right = np.matmul(np.concatenate((a_prev, xt), axis=-1).T, do)

            dbf_right = np.sum(df, axis=0, keepdims=True)
            dbc_right = np.sum(dpc, axis=0, keepdims=True)
            dbi_right = np.sum(di, axis=0, keepdims=True)
            dbo_right = np.sum(do, axis=0, keepdims=True)

            da_prev = (np.matmul(df, self.Wf_right[:self.n_a, :].T)
                        + np.matmul(di, self.Wi_right[:self.n_a, :].T)
                        + np.matmul(do, self.Wo_right[:self.n_a, :].T)
                        + np.matmul(dpc, self.Wc_right[:self.n_a, :].T))
            dc_prev = dC * f_gate + o_gate * (1 - np.tanh(c_next)**2) * f_gate * dA
            dxt = (np.matmul(df, self.Wf_right[self.n_a:, :].T)
                        + np.matmul(di, self.Wi_right[self.n_a:, :].T)
                        + np.matmul(do, self.Wo_right[self.n_a:, :].T)
                        + np.matmul(dpc, self.Wc_right[self.n_a:, :].T))
            return dxt, da_prev, dc_prev, dWc_right, dWf_right, dWi_right, dWo_right, dbc_right, dbf_right, dbi_right, dbo_right
        else:
            dWf_opp = np.matmul(np.concatenate((a_prev, xt), axis=-1).T, df)
            dWc_opp = np.matmul(np.concatenate((a_prev, xt), axis=-1).T, dpc)
            dWi_opp = np.matmul(np.concatenate((a_prev, xt), axis=-1).T, di)
            dWo_opp = np.matmul(np.concatenate((a_prev, xt), axis=-1).T, do)

            dbf_opp = np.sum(df, axis=0, keepdims=True)
            dbc_opp = np.sum(dpc, axis=0, keepdims=True)
            dbi_opp = np.sum(di, axis=0, keepdims=True)
            dbo_opp = np.sum(do, axis=0, keepdims=True)

            da_prev = (np.matmul(df, self.Wf_opp[:self.n_a, :].T)
                        + np.matmul(di, self.Wi_opp[:self.n_a, :].T)
                        + np.matmul(do, self.Wo_opp[:self.n_a, :].T)
                        + np.matmul(dpc, self.Wc_opp[:self.n_a, :].T))
            dc_prev = dC * f_gate + o_gate * (1 - np.tanh(c_next)**2) * f_gate * dA
            dxt = (np.matmul(df, self.Wf_opp[self.n_a:, :].T)
                        + np.matmul(di, self.Wi_opp[self.n_a:, :].T)
                        + np.matmul(do, self.Wo_opp[self.n_a:, :].T)
                        + np.matmul(dpc, self.Wc_opp[self.n_a:, :].T))
            return dxt, da_prev, dc_prev, dWc_opp, dWf_opp, dWi_opp, dWo_opp, dbc_opp, dbf_opp, dbi_opp, dbo_opp

    def backward(self, dA):
        self.dxt_caches = np.zeros_like(self.xt_caches)
        
        self.dWf_right = np.zeros_like(self.Wf_right)
        self.dbf_right = np.zeros_like(self.bf_right)
        self.dWi_right = np.zeros_like(self.Wi_right)
        self.dbi_right = np.zeros_like(self.bi_right)
        self.dWc_right = np.zeros_like(self.Wc_right)
        self.dbc_right = np.zeros_like(self.bc_right)
        self.dWo_right = np.zeros_like(self.Wo_right)
        self.dbo_right = np.zeros_like(self.bo_right)

        if self.return_sequences == False:
            if self.bidirectional == False:
                dA_right = np.zeros((self.batch_size, self.T_x, self.n_a))
                dA_right[:, -1, :] = dA
                dA_opp = None
            else:
                dA_right = np.zeros((self.batch_size, self.T_x, self.n_a))
                dA_opp = np.zeros((self.batch_size, self.T_x, self.n_a))
                if self.merge_mode == "concat":
                    dA_right[:, -1, :] = dA[:, :self.n_a]
                    dA_opp[:, 0, :] = dA[:, self.n_a:]
                elif self.merge_mode == "sum":
                    dA_right[:, -1, :] = dA
                    dA_opp[:, 0, :] = dA
                elif self.merge_mode == "average":
                    dA_right[:, -1, :] = dA / 2
                    dA_opp[:, 0, :] = dA / 2
                elif self.merge_mode == "multiply":
                    dA_right[:, -1, :] = dA * self.a_opp_caches[:, 0, :]
                    dA_opp[:, 0, :] = dA * self.a_right_caches[:, -1, :]
        else:
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
            
            self.dWf_opp = np.zeros_like(self.Wf_opp)
            self.dbf_opp = np.zeros_like(self.bf_opp)
            self.dWi_opp = np.zeros_like(self.Wi_opp)
            self.dbi_opp = np.zeros_like(self.bi_opp)
            self.dWc_opp = np.zeros_like(self.Wc_opp)
            self.dbc_opp = np.zeros_like(self.bc_opp)
            self.dWo_opp = np.zeros_like(self.Wo_opp)
            self.dbo_opp = np.zeros_like(self.bo_opp)

        da_prevt_right = np.zeros((self.batch_size, self.n_a))
        da_prevt_opp = np.zeros((self.batch_size, self.n_a))
        dc_prevt_right = np.zeros((self.batch_size, self.n_a))
        dc_prevt_opp = np.zeros((self.batch_size, self.n_a))

        for t in reversed(range(self.T_x)):
            xt = self.xt_caches[:,t,:]
            at_right = self.a_right_caches[:,t,:]
            ct_right = self.c_right_caches[:,t,:] 
            at_prev_right = self.a_prev_right_caches[:,t,:]
            ct_prev_right = self.c_prev_right_caches[:,t,:]

            c_hat_right = self.c_hat_right_caches[:,t,:]
            f_right = self.f_right_caches[:,t,:]
            i_right = self.i_right_caches[:,t,:]
            o_right = self.o_right_caches[:,t,:]

            dA_t_right = dA_right[:,t,:]

            dxt_right, da_prevt_right, dc_prevt_right, dWc_right, dWf_right, dWi_right, dWo_right, dbc_right, dbf_right, dbi_right, dbo_right = self.lstm_cell_backward(da_prevt_right + dA_t_right, xt, at_prev_right, dc_prevt_right, ct_right, c_hat_right, ct_prev_right, i_right, f_right, o_right, bidirectional=False)

            self.dWc_right += dWc_right
            self.dWf_right += dWf_right
            self.dWi_right += dWi_right
            self.dWo_right += dWo_right

            self.dbc_right += dbc_right
            self.dbf_right += dbf_right
            self.dbi_right += dbi_right
            self.dbo_right += dbo_right
            self.dxt_caches[:,t,:] += dxt_right
        self.da_0 = da_prevt_right
        self.dc_0 = dc_prevt_right

        if self.bidirectional:
            for t in range(self.T_x):
                xt = self.xt_caches[:,t,:]
                at_opp = self.a_opp_caches[:,t,:]
                ct_opp = self.c_opp_caches[:,t,:] 
                at_prev_opp = self.a_prev_opp_caches[:,t,:]
                ct_prev_opp = self.c_prev_opp_caches[:,t,:]
    
                c_hat_opp = self.c_hat_opp_caches[:,t,:]
                f_opp = self.f_opp_caches[:,t,:]
                i_opp = self.i_opp_caches[:,t,:]
                o_opp = self.o_opp_caches[:,t,:]
    
                dA_t_opp = dA_opp[:,t,:]
    
                dxt_opp, da_prevt_opp, dc_prevt_opp, dWc_opp, dWf_opp, dWi_opp, dWo_opp, dbc_opp, dbf_opp, dbi_opp, dbo_opp = self.lstm_cell_backward(da_prevt_opp + dA_t_opp, xt, at_prev_opp, dc_prevt_opp, ct_opp, c_hat_opp, ct_prev_opp, i_opp, f_opp, o_opp, bidirectional=True)
    
                self.dWc_opp += dWc_opp
                self.dWf_opp += dWf_opp
                self.dWi_opp += dWi_opp
                self.dWo_opp += dWo_opp
    
                self.dbc_opp += dbc_opp
                self.dbf_opp += dbf_opp
                self.dbi_opp += dbi_opp
                self.dbo_opp += dbo_opp
                self.dxt_caches[:,t,:] += dxt_opp
            self.da_0_opp = da_prevt_opp
            self.dc_0_opp = dc_prevt_opp

        m = self.batch_size
        
        if self.regularizer == "l1":
            self.dWc_right += (self.lambd / m) * np.sign(self.Wc_right)
            self.dWf_right += (self.lambd / m) * np.sign(self.Wf_right)
            self.dWi_right += (self.lambd / m) * np.sign(self.Wi_right)
            self.dWo_right += (self.lambd / m) * np.sign(self.Wo_right)
            if self.bidirectional:
                self.dWc_opp += (self.lambd / m) * np.sign(self.Wc_opp)
                self.dWf_opp += (self.lambd / m) * np.sign(self.Wf_opp)
                self.dWi_opp += (self.lambd / m) * np.sign(self.Wi_opp)
                self.dWo_opp += (self.lambd / m) * np.sign(self.Wo_opp)
        elif self.regularizer == "l2":
            self.dWc_right += (self.lambd / m) * self.Wc_right
            self.dWf_right += (self.lambd / m) * self.Wf_right
            self.dWi_right += (self.lambd / m) * self.Wi_right
            self.dWo_right += (self.lambd / m) * self.Wo_right
            if self.bidirectional:
                self.dWc_opp += (self.lambd / m) * self.Wc_opp
                self.dWf_opp += (self.lambd / m) * self.Wf_opp
                self.dWi_opp += (self.lambd / m) * self.Wi_opp
                self.dWo_opp += (self.lambd / m) * self.Wo_opp
        
        gradients = {
            "dx": self.dxt_caches,
            "dWc": self.dWc_right,
            "dWf": self.dWf_right,
            "dWi": self.dWi_right,
            "dWo": self.dWo_right,
            "dbc": self.dbc_right,
            "dbf": self.dbf_right,
            "dbi": self.dbi_right,
            "dbo": self.dbo_right
        }
        return self.dxt_caches

    def update_parameters(self, learning_rate = 0.01, optimizer=None, beta1 = 0.9, beta2 = 0.99, maxValue = None, minValue = None):
            if self.freeze:
                return

            if optimizer == "rmsprop":
                self.t += 1
                Wc_right_update, bc_right_update, self.Sc_right = rmsprop(self.dWc_right, self.dbc_right, self.Sc_right, self.t, beta1)
                Wf_right_update, bf_right_update, self.Sf_right = rmsprop(self.dWf_right, self.dbf_right, self.Sf_right, self.t, beta1)
                Wi_right_update, bi_right_update, self.Si_right = rmsprop(self.dWi_right, self.dbi_right, self.Si_right, self.t, beta1)
                Wo_right_update, bo_right_update, self.So_right = rmsprop(self.dWo_right, self.dbo_right, self.So_right, self.t, beta1)
                if self.bidirectional:
                    Wc_opp_update, bc_opp_update, self.Sc_opp = rmsprop(self.dWc_opp, self.dbc_opp, self.Sc_opp, self.t, beta1)
                    Wf_opp_update, bf_opp_update, self.Sf_opp = rmsprop(self.dWf_opp, self.dbf_opp, self.Sf_opp, self.t, beta1)
                    Wi_opp_update, bi_opp_update, self.Si_opp = rmsprop(self.dWi_opp, self.dbi_opp, self.Si_opp, self.t, beta1)
                    Wo_opp_update, bo_opp_update, self.So_opp = rmsprop(self.dWo_opp, self.dbo_opp, self.So_opp, self.t, beta1)
            elif optimizer == "momentum":
                self.t += 1
                Wc_right_update, bc_right_update, self.Sc_right = momentum(self.dWc_right, self.dbc_right, self.Sc_right, self.t, beta1)
                Wf_right_update, bf_right_update, self.Sf_right = momentum(self.dWf_right, self.dbf_right, self.Sf_right, self.t, beta1)
                Wi_right_update, bi_right_update, self.Si_right = momentum(self.dWi_right, self.dbi_right, self.Si_right, self.t, beta1)
                Wo_right_update, bo_right_update, self.So_right = momentum(self.dWo_right, self.dbo_right, self.So_right, self.t, beta1)
                if self.bidirectional:
                    Wc_opp_update, bc_opp_update, self.Sc_opp = momentum(self.dWc_opp, self.dbc_opp, self.Sc_opp, self.t, beta1)
                    Wf_opp_update, bf_opp_update, self.Sf_opp = momentum(self.dWf_opp, self.dbf_opp, self.Sf_opp, self.t, beta1)
                    Wi_opp_update, bi_opp_update, self.Si_opp = momentum(self.dWi_opp, self.dbi_opp, self.Si_opp, self.t, beta1)
                    Wo_opp_update, bo_opp_update, self.So_opp = momentum(self.dWo_opp, self.dbo_opp, self.So_opp, self.t, beta1)
            elif optimizer == "adam":
                self.t += 1
                Wc_right_update, bc_right_update, self.Vc_right, self.Sc_right, _, _ = adam(self.dWc_right, self.dbc_right, self.Vc_right, self.Sc_right, self.t, beta1, beta2)
                Wf_right_update, bf_right_update, self.Vf_right, self.Sf_right, _, _ = adam(self.dWf_right, self.dbf_right, self.Vf_right, self.Sf_right, self.t, beta1, beta2)
                Wi_right_update, bi_right_update, self.Vi_right, self.Si_right, _, _ = adam(self.dWi_right, self.dbi_right, self.Vi_right, self.Si_right, self.t, beta1, beta2)
                Wo_right_update, bo_right_update, self.Vo_right, self.So_right, _, _ = adam(self.dWo_right, self.dbo_right, self.Vo_right, self.So_right, self.t, beta1, beta2)
                if self.bidirectional:
                    Wc_opp_update, bc_opp_update, self.Vc_opp, self.Sc_opp, _, _ = adam(self.dWc_opp, self.dbc_opp, self.Vc_opp, self.Sc_opp, self.t, beta1, beta2)
                    Wf_opp_update, bf_opp_update, self.Vf_opp, self.Sf_opp, _, _ = adam(self.dWf_opp, self.dbf_opp, self.Vf_opp, self.Sf_opp, self.t, beta1, beta2)
                    Wi_opp_update, bi_opp_update, self.Vi_opp, self.Si_opp, _, _ = adam(self.dWi_opp, self.dbi_opp, self.Vi_opp, self.Si_opp, self.t, beta1, beta2)
                    Wo_opp_update, bo_opp_update, self.Vo_opp, self.So_opp, _, _ = adam(self.dWo_opp, self.dbo_opp, self.Vo_opp, self.So_opp, self.t, beta1, beta2)
            else:
                Wc_right_update = self.dWc_right
                Wf_right_update = self.dWf_right
                Wi_right_update = self.dWi_right
                Wo_right_update = self.dWo_right

                bc_right_update = self.dbc_right
                bf_right_update = self.dbf_right
                bi_right_update = self.dbi_right
                bo_right_update = self.dbo_right
                if self.bidirectional:
                    Wc_opp_update = self.dWc_opp
                    Wf_opp_update = self.dWf_opp
                    Wi_opp_update = self.dWi_opp
                    Wo_opp_update = self.dWo_opp
    
                    bc_opp_update = self.dbc_opp
                    bf_opp_update = self.dbf_opp
                    bi_opp_update = self.dbi_opp
                    bo_opp_update = self.dbo_opp
            
            if maxValue or minValue is not None:
                Wc_right_update = np.clip(Wc_right_update, minValue, maxValue)
                Wf_right_update = np.clip(Wf_right_update, minValue, maxValue)
                Wi_right_update = np.clip(Wi_right_update, minValue, maxValue)
                Wo_right_update = np.clip(Wo_right_update, minValue, maxValue)

                bc_right_update = np.clip(bc_right_update, minValue, maxValue)
                bf_right_update = np.clip(bf_right_update, minValue, maxValue)
                bi_right_update = np.clip(bi_right_update, minValue, maxValue)
                bo_right_update = np.clip(bo_right_update, minValue, maxValue)
                if self.bidirectional:
                    Wc_opp_update = np.clip(Wc_opp_update, minValue, maxValue)
                    Wf_opp_update = np.clip(Wf_opp_update, minValue, maxValue)
                    Wi_opp_update = np.clip(Wi_opp_update, minValue, maxValue)
                    Wo_opp_update = np.clip(Wo_opp_update, minValue, maxValue)
    
                    bc_opp_update = np.clip(bc_opp_update, minValue, maxValue)
                    bf_opp_update = np.clip(bf_opp_update, minValue, maxValue)
                    bi_opp_update = np.clip(bi_opp_update, minValue, maxValue)
                    bo_opp_update = np.clip(bo_opp_update, minValue, maxValue)
            
            self.Wc_right -= learning_rate * Wc_right_update
            self.Wf_right -= learning_rate * Wf_right_update
            self.Wi_right -= learning_rate * Wi_right_update
            self.Wo_right -= learning_rate * Wo_right_update

            self.bc_right -= learning_rate * bc_right_update
            self.bf_right -= learning_rate * bf_right_update
            self.bi_right -= learning_rate * bi_right_update
            self.bo_right -= learning_rate * bo_right_update
            if self.bidirectional:
                self.Wc_opp -= learning_rate * Wc_opp_update
                self.Wf_opp -= learning_rate * Wf_opp_update
                self.Wi_opp -= learning_rate * Wi_opp_update
                self.Wo_opp -= learning_rate * Wo_opp_update
    
                self.bc_opp -= learning_rate * bc_opp_update
                self.bf_opp -= learning_rate * bf_opp_update
                self.bi_opp -= learning_rate * bi_opp_update
                self.bo_opp -= learning_rate * bo_opp_update

    def get_regularization_penalty(self, m):
        pen = 0.0
        if self.regularizer == "l1":
            pen += (self.lambd / m) * np.sum(np.abs(self.Wc_right))
            pen += (self.lambd / m) * np.sum(np.abs(self.Wf_right))
            pen += (self.lambd / m) * np.sum(np.abs(self.Wi_right))
            pen += (self.lambd / m) * np.sum(np.abs(self.Wo_right))
            if self.bidirectional:
                pen += (self.lambd / m) * np.sum(np.abs(self.Wc_opp))
                pen += (self.lambd / m) * np.sum(np.abs(self.Wf_opp))
                pen += (self.lambd / m) * np.sum(np.abs(self.Wi_opp))
                pen += (self.lambd / m) * np.sum(np.abs(self.Wo_opp))
        elif self.regularizer == "l2":
            pen += (self.lambd / (2 * m)) * np.sum(np.square(self.Wc_right))
            pen += (self.lambd / (2 * m)) * np.sum(np.square(self.Wf_right))
            pen += (self.lambd / (2 * m)) * np.sum(np.square(self.Wi_right))
            pen += (self.lambd / (2 * m)) * np.sum(np.square(self.Wo_right))
            if self.bidirectional:
                pen += (self.lambd / (2 * m)) * np.sum(np.square(self.Wc_opp))
                pen += (self.lambd / (2 * m)) * np.sum(np.square(self.Wf_opp))
                pen += (self.lambd / (2 * m)) * np.sum(np.square(self.Wi_opp))
                pen += (self.lambd / (2 * m)) * np.sum(np.square(self.Wo_opp))
        else:
            pen = 0.0
        return pen

import numpy as np
from functions.utilities import random_mini_batch
from functions.loss import categorical_cross_entropy, mean_square_error, binary_cross_entropy

class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        
        self.layers = []
        self.attention = None
        
        self.learning_rate = 0.001
        self.cost_func = "categorical_cross_entropy"
        self.optimizer = None
        self.beta1 = 0.9
        self.beta2 = 0.99
        
        self.enc_lstms = [l for l in self.encoder.layers if hasattr(l, 'lstm_cell_forward')]
        self.dec_lstms = [l for l in self.decoder.layers if hasattr(l, 'lstm_cell_forward')]
        
        self.pre_lstm_layers = []
        self.step_layers = []
        
        found_first_lstm = False
        for layer in self.decoder.layers:
            if hasattr(layer, 'lstm_cell_forward'):
                found_first_lstm = True
            if found_first_lstm:
                self.step_layers.append(layer)
            else:
                self.pre_lstm_layers.append(layer)
                
        if self.dec_lstms:
            last_lstm = self.dec_lstms[-1]
            self.current_layer_neurons = last_lstm.n_a
            if getattr(last_lstm, 'bidirectional', False) and getattr(last_lstm, 'merge_mode', 'concat') == "concat":
                self.current_layer_neurons = last_lstm.n_a * 2

    def add(self, layer):
        if type(layer).__name__ == "Attention":
            self.attention = layer
        else:
            layer.init_params(self.current_layer_neurons)
            if hasattr(layer, 'number_neurons'):
                self.current_layer_neurons = layer.number_neurons
        self.layers.append(layer)

    def init_optimizer(self, optimizer):
        self.encoder.init_layer_optimizer(optimizer)
        self.decoder.init_layer_optimizer(optimizer)
        for layer in self.layers:
            if hasattr(layer, 'init_optimizer'):
                layer.init_optimizer(optimizer)

    def compute_cost(self, Y_true, Y_pred, derivative=False):
        if self.cost_func == "categorical_cross_entropy":
            return categorical_cross_entropy(Y_true, Y_pred, derivative)

    def forward(self, X, Y, training=True):
        self.X = X
        self.Y = Y
        
        self.enc_out, _ = self.encoder.forward(X, training=training)
        
        for enc_l, dec_l in zip(self.enc_lstms, self.dec_lstms):
            enc_l.pass_states_to(dec_l)
            
        A_dec = Y
        for layer in self.pre_lstm_layers:
            A_dec = layer.forward(A_dec, training=training)
        self.dec_emb_out = A_dec
        
        batch_size = Y.shape[0]
        T_y = Y.shape[1]
        
        if training:
            self.step_caches = [[] for _ in range(T_y)]
            if self.attention:
                self.attention.reset_caches()
                
        out = []
        first_lstm = self.dec_lstms[0]
        
        for t in range(T_y):
            y_t = self.dec_emb_out[:, t, :]
            s_prev = first_lstm.a_state if first_lstm.a_state is not None else np.zeros((batch_size, first_lstm.n_a))
            
            if self.attention:
                ctx_t, _ = self.attention.forward(s_prev, self.enc_out, y_t, training=training)
            else:
                ctx_t = y_t
                
            a_l = ctx_t
            
            for layer_idx, layer in enumerate(self.step_layers):
                if hasattr(layer, 'lstm_cell_forward'):
                    s_l = layer.a_state if layer.a_state is not None else np.zeros((batch_size, layer.n_a))
                    c_l = layer.c_state if layer.c_state is not None else np.zeros((batch_size, layer.n_a))
                    
                    s_next, c_next, c_hat, o_gate, i_gate, f_gate = layer.lstm_cell_forward(a_l, s_l, c_l)
                    
                    if training:
                        self.step_caches[t].append((layer_idx, 'lstm', a_l, s_l, c_l, c_next, c_hat, o_gate, i_gate, f_gate))
                        
                    layer.a_state, layer.c_state = s_next, c_next
                    a_l = s_next
                else:
                    a_l = layer.forward(a_l, training=training)
                    if training:
                        cache_dict = {}
                        attrs_to_cache = ['activation_cache', 'batch_mean', 'batch_var', 'A_norm', 'mask', 'A_prev', 'cache']
                        for attr in attrs_to_cache:
                            if hasattr(layer, attr):
                                cache_dict[attr] = getattr(layer, attr)
                        self.step_caches[t].append((layer_idx, 'other', cache_dict))
                        
            out.append(a_l)
            
        self.dec_step_out = np.stack(out, axis=1)
        
        AL = self.dec_step_out
        for layer in self.layers:
            if type(layer).__name__ == "Attention": continue
            AL = layer.forward(AL, training=training)
            
        return AL

    def backward(self, dL):
        dA = dL
        for layer in reversed(self.layers):
            if type(layer).__name__ == "Attention": continue
            dA = layer.backward(dA)
            layer.update_parameters(self.learning_rate, self.optimizer, self.beta1, self.beta2)
            
        dA_step_out = dA
        batch_size, T_y, _ = dA_step_out.shape
        
        for dl in self.dec_lstms:
            dl.dWf_right, dl.dbf_right = np.zeros_like(dl.Wf_right), np.zeros_like(dl.bf_right)
            dl.dWi_right, dl.dbi_right = np.zeros_like(dl.Wi_right), np.zeros_like(dl.bi_right)
            dl.dWc_right, dl.dbc_right = np.zeros_like(dl.Wc_right), np.zeros_like(dl.bc_right)
            dl.dWo_right, dl.dbo_right = np.zeros_like(dl.Wo_right), np.zeros_like(dl.bo_right)
            
        if self.attention and getattr(self.attention, 'mode', 'dot') == "dense":
            self.attention.dense_concat.dW, self.attention.dense_concat.db = np.zeros_like(self.attention.dense_concat.W), np.zeros_like(self.attention.dense_concat.b)
            self.attention.dense_score.dW, self.attention.dense_score.db = np.zeros_like(self.attention.dense_score.W), np.zeros_like(self.attention.dense_score.b)
            
        shared_grads = {}
        for idx, layer in enumerate(self.step_layers):
            if not hasattr(layer, 'lstm_cell_forward'):
                shared_grads[idx] = {}
                if hasattr(layer, 'W'):
                    shared_grads[idx]['dW'], shared_grads[idx]['db'] = np.zeros_like(layer.W), np.zeros_like(layer.b)
                if hasattr(layer, 'gamma'):
                    shared_grads[idx]['d_gamma'], shared_grads[idx]['d_beta'] = np.zeros_like(layer.gamma), np.zeros_like(layer.beta)

        da_next = {idx: np.zeros((batch_size, l.n_a)) for idx, l in enumerate(self.step_layers) if hasattr(l, 'lstm_cell_forward')}
        dc_next = {idx: np.zeros((batch_size, l.n_a)) for idx, l in enumerate(self.step_layers) if hasattr(l, 'lstm_cell_forward')}
        first_lstm_idx = next(i for i, l in enumerate(self.step_layers) if hasattr(l, 'lstm_cell_forward'))
        
        d_enc_out = np.zeros_like(self.enc_out)
        d_dec_emb = np.zeros_like(self.dec_emb_out)
        
        for t in reversed(range(T_y)):
            dA_t = dA_step_out[:, t, :]
            t_caches = self.step_caches[t]
            
            for cache in reversed(t_caches):
                layer_idx, l_type = cache[0], cache[1]
                layer = self.step_layers[layer_idx]
                
                if l_type == 'lstm':
                    _, _, xt, a_prev, c_prev, c_next, c_hat, o_gate, i_gate, f_gate = cache
                    dA_combined = dA_t + da_next[layer_idx]
                    
                    dxt, da_prev_t, dc_prev_t, dWc, dWf, dWi, dWo, dbc, dbf, dbi, dbo = layer.lstm_cell_backward(
                        dA_combined, xt, a_prev, dc_next[layer_idx], c_next, c_hat, c_prev, i_gate, f_gate, o_gate, bidirectional=False)
                        
                    layer.dWc_right += dWc; layer.dWf_right += dWf; layer.dWi_right += dWi; layer.dWo_right += dWo
                    layer.dbc_right += dbc; layer.dbf_right += dbf; layer.dbi_right += dbi; layer.dbo_right += dbo
                    
                    da_next[layer_idx], dc_next[layer_idx] = da_prev_t, dc_prev_t
                    dA_t = dxt
                else:
                    cache_dict = cache[2]
                    for k, v in cache_dict.items():
                        setattr(layer, k, v)
                    
                    dA_t = layer.backward(dA_t)
                    
                    if hasattr(layer, 'dW'):
                        shared_grads[layer_idx]['dW'] += layer.dW
                        shared_grads[layer_idx]['db'] += layer.db
                    if hasattr(layer, 'd_gamma'):
                        shared_grads[layer_idx]['d_gamma'] += layer.d_gamma
                        shared_grads[layer_idx]['d_beta'] += layer.d_beta
                    
            if self.attention:
                d_s_prev, d_h, d_y_emb = self.attention.backward_step(dA_t)
                da_next[first_lstm_idx] += d_s_prev
                d_enc_out += d_h
                if d_y_emb is not None: d_dec_emb[:, t, :] += d_y_emb
            else:
                d_dec_emb[:, t, :] += dA_t
            
        for idx, layer in enumerate(self.step_layers):
            if idx in shared_grads:
                if 'dW' in shared_grads[idx]:
                    layer.dW, layer.db = shared_grads[idx]['dW'], shared_grads[idx]['db']
                if 'd_gamma' in shared_grads[idx]:
                    layer.d_gamma, layer.d_beta = shared_grads[idx]['d_gamma'], shared_grads[idx]['d_beta']

        if self.attention:
            self.attention.update_parameters(self.learning_rate, self.optimizer, self.beta1, self.beta2)
        for layer in self.step_layers:
            if hasattr(layer, 'update_parameters'):
                layer.update_parameters(self.learning_rate, self.optimizer, self.beta1, self.beta2)
                
        dA_dec_front = d_dec_emb
        for layer in reversed(self.pre_lstm_layers):
            dA_dec_front = layer.backward(dA_dec_front)
            if hasattr(layer, 'update_parameters'):
                layer.update_parameters(self.learning_rate, self.optimizer, self.beta1, self.beta2)
                
        for dec_lstm_layer, enc_lstm_layer in zip(self.dec_lstms, self.enc_lstms):
            layer_idx = self.step_layers.index(dec_lstm_layer)
            da_decoder = da_next[layer_idx]
            dc_decoder = dc_next[layer_idx]
            
            enc_lstm_layer.set_gradient_from_decoder(da_decoder, dc_decoder)

        self.encoder.learning_rate, self.encoder.optimizer = self.learning_rate, self.optimizer
        self.encoder.beta1, self.encoder.beta2 = self.beta1, self.beta2
        
        d_enc_in = self.encoder.backward(d_enc_out)
        
        return d_enc_in

    def fit(self, training_set, num_epochs, optimizer=None, learning_rate=0.001, beta1=0.9, beta2=0.99, verbose=True):
        X_train, Y_train_input, Y_train_target = training_set
        self.optimizer, self.learning_rate = optimizer, learning_rate
        self.beta1, self.beta2 = beta1, beta2
        
        if optimizer is not None:
            self.init_optimizer(optimizer)
            
        for epoch in range(num_epochs):
            AL = self.forward(X_train, Y_train_input)
            cost = self.compute_cost(Y_train_target, AL)
            
            dL = self.compute_cost(Y_train_target, AL, derivative=True)
            self.backward(dL)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {cost:.4f}")
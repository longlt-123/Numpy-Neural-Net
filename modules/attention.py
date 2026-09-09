import numpy as np
from modules.base import Layer
from modules.dense_layer import Dense
from modules.repeat_vector import RepeatVector
from functions.output import softmax

class Attention(Layer):
    def __init__(self, mode="dot", context_mode="concat"):
        self.mode = mode
        self.context_mode = context_mode
        self.dense_concat = None
        self.dense_score = None
        self.reset_caches()

    def init_params(self, s_dim, h_dim):
        if self.mode == "dense":
            self.dense_concat = Dense(h_dim, activation="linear")
            self.dense_concat.init_params(s_dim + h_dim)
            self.dense_score = Dense(1, activation="linear")
            self.dense_score.init_params(h_dim)

    def init_optimizer(self, optimizer):
        if self.mode == "dense":
            self.dense_concat.init_optimizer(optimizer)
            self.dense_score.init_optimizer(optimizer)

    def update_parameters(self, learning_rate, optimizer, beta1, beta2):
        if self.mode == "dense":
            self.dense_concat.update_parameters(learning_rate, optimizer, beta1, beta2)
            self.dense_score.update_parameters(learning_rate, optimizer, beta1, beta2)

    def reset_caches(self):
        self.s_prev_caches = []
        self.h_caches = []
        self.y_emb_caches = []
        self.alpha_caches = []
        self.repeat_caches = []
        if self.mode == "dense":
            self.dense1_caches = []
            self.dense2_caches = []
            self.x_tanh_caches = []

    def forward(self, s_prev, h, y_emb=None, training=True):
        batch_size, seq_len, _ = h.shape
        
        repeat_layer = RepeatVector(seq_len, axis=1)
        s_expanded = repeat_layer.forward(s_prev)
        
        if training:
            self.repeat_caches.append(repeat_layer)
        
        if self.mode == "dot":
            score = np.sum(s_expanded * h, axis=-1)
        elif self.mode == "dense":
            concat_input = np.concatenate([s_expanded, h], axis=-1)
            x, cache1 = self.dense_concat.forward_propagation(concat_input, self.dense_concat.W, self.dense_concat.b, self.dense_concat.activation)
            x_tanh = np.tanh(x)
            score_out, cache2 = self.dense_score.forward_propagation(x_tanh, self.dense_score.W, self.dense_score.b, self.dense_score.activation)
            score = np.squeeze(score_out, axis=-1)
            
            if training:
                self.dense1_caches.append(cache1)
                self.dense2_caches.append(cache2)
                self.x_tanh_caches.append(x_tanh)

        alpha = softmax(score)
        alpha_expanded = np.expand_dims(alpha, axis=-1)
        context = np.sum(alpha_expanded * h, axis=1)

        if training:
            self.s_prev_caches.append(s_prev)
            self.h_caches.append(h)
            self.y_emb_caches.append(y_emb)
            self.alpha_caches.append(alpha)
            
        if self.context_mode == "direct":
            return context, alpha
        elif self.context_mode == "concat":
            output = np.concatenate([y_emb, context], axis=-1)
            return output, alpha

        return context, alpha

    def backward(self, d_output):
        s_prev = self.s_prev_caches.pop()
        h = self.h_caches.pop()
        y_emb = self.y_emb_caches.pop()
        alpha = self.alpha_caches.pop()
        repeat_layer = self.repeat_caches.pop()
        
        if self.context_mode == "direct":
            d_context = d_output
            d_y_emb = None
        elif self.context_mode == "concat":
            y_emb_dim = y_emb.shape[-1]
            d_y_emb = d_output[:, :y_emb_dim]
            d_context = d_output[:, y_emb_dim:]
            
        d_alpha = np.sum(d_context[:, np.newaxis, :] * h, axis=-1)
        d_h = d_context[:, np.newaxis, :] * np.expand_dims(alpha, axis=-1)
        
        sum_alpha_d_alpha = np.sum(alpha * d_alpha, axis=-1, keepdims=True)
        d_score = alpha * (d_alpha - sum_alpha_d_alpha)
        
        if self.mode == "dot":
            d_s_expanded = d_score[:, :, np.newaxis] * h
            
            s_expanded = repeat_layer.forward(s_prev)
            d_h += d_score[:, :, np.newaxis] * s_expanded
            d_s_prev = repeat_layer.backward(d_s_expanded)
            
        elif self.mode == "dense":
            cache2 = self.dense2_caches.pop()
            cache1 = self.dense1_caches.pop()
            x_tanh = self.x_tanh_caches.pop()
            
            d_score_expanded = np.expand_dims(d_score, axis=-1)
            d_x_tanh, dW2, db2 = self.dense_score.backward_propagation(d_score_expanded, cache2, self.dense_score.activation)
            self.dense_score.dW += dW2
            self.dense_score.db += db2
            
            d_x = d_x_tanh * (1 - np.square(x_tanh))
            d_concat, dW1, db1 = self.dense_concat.backward_propagation(d_x, cache1, self.dense_concat.activation)
            self.dense_concat.dW += dW1
            self.dense_concat.db += db1
            
            s_dim = s_prev.shape[-1]
            d_s_expanded = d_concat[:, :, :s_dim]
            d_h += d_concat[:, :, s_dim:]
            
            d_s_prev = repeat_layer.backward(d_s_expanded)
            
        return d_s_prev, d_h, d_y_emb
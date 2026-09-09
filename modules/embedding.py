import numpy as np
from modules.base import Layer
from functions.utilities import initialize_parameters, initialize_optimizer

class Embedding(Layer):
    def __init__(self, embedding_dim, init_type = "he", input_type="idx", index_column = 0, transfer_weights = None, terminal_word = None, freeze = True):
        self.embedding_dim = embedding_dim
        self.init_type = init_type
        self.input_type = input_type
        self.index_column = index_column
        self.terminal_word = terminal_word
        self.freeze = freeze
        self.A_prev = None
        self.E = None
        if transfer_weights is not None:
            if self.terminal_word is None:
                self.E = transfer_weights
            else:
                old_E = transfer_weights
                vocab_size = old_E.shape[0]
                self.E = np.zeros(
                    (vocab_size + len(self.terminal_word), self.embedding_dim)
                )
                self.E[:vocab_size, :] = transfer_weights

                for i in range(len(self.terminal_word)):
                    if self.terminal_word[i] == "<PAD>":
                        self.E[vocab_size + i] = np.zeros(self.embedding_dim)
                    else:
                        self.E[vocab_size + i] = np.random.randn(self.embedding_dim) * np.sqrt(2 / vocab_size)
        self.dE = None

    def init_params(self, vocab_size=None):
        if self.terminal_word is not None:
            total_vocab_size = vocab_size + len(self.terminal_word)
        else:
            total_vocab_size = vocab_size
        self.E = initialize_parameters(total_vocab_size, self.embedding_dim, self.init_type)
        if self.terminal_word is not None:
            for i in range(len(self.terminal_word)):
                if self.terminal_word[i] == "<PAD>":
                    self.E[vocab_size + i] = np.zeros(self.embedding_dim)
                else:
                    self.E[vocab_size + i] = np.random.randn(self.embedding_dim) * np.sqrt(2 / vocab_size)

    def forward(self, A_prev, training=False):
        if self.input_type == "idx":
            A = self.E[A_prev]
        elif self.input_type == "one-hot":
            A = np.matmul(A_prev, self.E)
        elif self.input_type == "single_idx":
            A = self.E[A_prev[:, self.index_column]]
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

        self.A_prev = A_prev
        return A

    def backward(self, dA):
        self.dE = np.zeros_like(self.E)
        if self.input_type == "idx":
            np.add.at(
                self.dE,
                self.A_prev,
                dA
            )
            dA_prev = None
        elif self.input_type == "one-hot":
            self.dE = np.einsum(
                'mkv,mkd->vd',
                self.A_prev,
                dA
            )
            dA_prev = np.matmul(
                dA,
                self.E.T
            )
        elif self.input_type == "single_idx":
            token_ids = self.A_prev[:, self.index_column]
            np.add.at(
                self.dE,
                token_ids,
                dA
            )
            dA_prev = None
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")

        return dA_prev

    def update_parameters(self, learning_rate=0.01, optimizer=None, maxValue=None, minValue=None):
        if self.freeze:
            return
        self.E -= learning_rate * self.dE

    def set_weights(self, weights):
        self.E = weights
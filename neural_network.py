import numpy as np

from functions.activations import relu, linear, sigmoid
from functions.output import softmax
from functions.loss import mean_square_error, categorical_cross_entropy, binary_cross_entropy
from functions.score import accuracy_score, precision_score, recall_score, f1_score
from functions.utilities import forward_propagation, backward_propagation, convert_targets, initialize_parameters, initialize_optimizer, shuffle_data

from modules.activation_layer import Activation
from modules.dense_layer import Dense
from modules.batchnorm import BatchNorm
from modules.dropout import Dropout

from optimizers.adam import adam
from optimizers.learning_rate_decay import decay_with_decay_rate, decay_with_stage
from optimizers.momentum import momentum
from optimizers.RMSprop import rmsprop

class NeuralNetwork():
    def __init__(self, layers):
        self.layers: list = []
        for layer in layers:
            self.layers.append(layer)

    def add(self, layer):
        self.layers.append(layer)

    def random_mini_batch(X, Y, mini_batch_size = 64, seed = 0):
        m = X.shape[0]
        shuffle_X, shuffle_Y = shuffle_data(X, Y)

        mini_batches = []

        num_complete_minibatches = np.floor(m / mini_batch_size)
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffle_X[k * mini_batch_size : (k+1) * mini_batch_size, :]
            mini_batch_Y = shuffle_Y[k * mini_batch_size : (k+1) * mini_batch_size, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            mini_batch_X = shuffle_X[num_complete_minibatches * mini_batch_size :, :]
            mini_batch_Y = shuffle_Y[num_complete_minibatches * mini_batch_size :, :]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches
import numpy as np

from functions.activations import relu, linear, sigmoid
from functions.output import softmax
from functions.loss import mean_square_error, categorical_cross_entropy, binary_cross_entropy
from functions.score import accuracy_score, precision_score, recall_score, f1_score
from functions.utilities import forward_propagation, backward_propagation, convert_targets, initialize_parameters, initialize_optimizer, shuffle_data

from modules.base import Layer
from modules.activation_layer import Activation
from modules.dense_layer import Dense
from modules.batchnorm import BatchNorm
from modules.dropout import Dropout

from optimizers.adam import adam
from optimizers.learning_rate_decay import decay_with_decay_rate, decay_with_stage
from optimizers.momentum import momentum
from optimizers.RMSprop import rmsprop

class NeuralNetwork():
    def __init__(self, input_dim, layers: list[Layer] = None):
        self.layers: list = []
        self.optimizer = None
        self.learning_rate = 0
        self.cost_func = None
        self.regularize_penalty = 0

        if isinstance(input_dim, tuple):
            self.current_layer_neurons = input_dim[1]
        else:
            self.current_layer_neurons = input_dim

        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if isinstance(layer, Activation) or isinstance(layer, Dropout):
            pass
        else:
            layer.init_params(self.current_layer_neurons)
            self.current_layer_neurons = layer.number_neurons
        self.layers.append(layer)

    @staticmethod
    def random_mini_batch(X, Y, mini_batch_size = 64, seed = 0):
        if seed != 0:
            np.random.seed(seed)
        
        m = X.shape[0]
        shuffle_X, shuffle_Y = shuffle_data(X, Y)

        mini_batches = []

        num_complete_minibatches = m // mini_batch_size
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

    def forward(self, X):
        AL = X
        regularize_penalty = 0
        for layer in self.layers:
            A_prev = AL
            AL = layer.forward(A_prev)
            regularize_penalty += layer.get_regularization_penalty(X.shape[0])
        return AL, regularize_penalty

    def backward(self, dL):
        dA_prev = dL
        for layer in reversed(self.layers):
            dA = dA_prev
            dA_prev = layer.backward(dA)
            layer.update_parameters(self.learning_rate, self.optimizer)

    def compute_cost(self, Y_true, Y_pred, derivative = False):
        if self.cost_func == "mse":
            if derivative:
                return mean_square_error(Y_true, Y_pred, derivative)
            return mean_square_error(Y_true, Y_pred) + self.regularize_penalty
        
        elif self.cost_func == "binary_cross_entropy":
            if derivative:
                return binary_cross_entropy(Y_true, Y_pred, derivative)
            return binary_cross_entropy(Y_true, Y_pred) + self.regularize_penalty
        
        elif self.cost_func == "categorical_cross_entropy":
            if derivative:
                return categorical_cross_entropy(Y_true, Y_pred, derivative)
            return categorical_cross_entropy(Y_true, Y_pred) + self.regularize_penalty
        
        else:
            raise ValueError("Invalid cost function.")

    def compute_score(self, Y_true, Y_pred, type):
        if type == "accuracy":
            return accuracy_score(Y_true, Y_pred)
        elif type == "precision":
            return precision_score(Y_true, Y_pred)
        elif type == "recall":
            return recall_score(Y_true, Y_pred)
        elif type == "f1":
            return f1_score(Y_true, Y_pred)
        else:
            raise ValueError("Invalid score type")

    def init_layer_optimizer(self, optimizer):
        for layer in self.layers:
            layer.init_optimizer(optimizer)

    def fit(self, training_set, validation_set = None, num_epochs = None, cost_function = None, optimizer = None, learning_rate = 0.001, mini_batch_size = 64, beta1 = 0.9, beta2 = 0.99, 
            epsilon = 1e-8, decay = None, decay_rate = 1, verbose = True):
        X_train, Y_train = training_set
        if validation_set is not None:
            X_val, Y_val = validation_set

        self.cost_func = cost_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        training_costs = []
        validation_costs = []
        learning_rates = []
        t = 0
        m = X_train.shape[0]
        seed = 10

        if optimizer is not None:
            self.init_layer_optimizer(optimizer)

        for epoch in range(0, num_epochs):
            seed = seed + 1
            training_mini_batches = self.random_mini_batch(X_train, Y_train, mini_batch_size, seed)
            total_training_cost = 0
            num_train_batches = len(training_mini_batches)

            if verbose:
                print(f"\n[{'-'*15} EPOCH {epoch + 1}/{num_epochs} {'-'*15}]")

            for batch_idx, mini_batch in enumerate(training_mini_batches):
                mini_batch_X, mini_batch_Y = mini_batch

                AL, self.regularize_penalty = self.forward(mini_batch_X)
                batch_train_cost = self.compute_cost(mini_batch_Y, AL)
                total_training_cost += batch_train_cost

                dL = self.compute_cost(mini_batch_Y, AL, derivative=True)
                self.backward(dL)

                if verbose:
                    print(f"Training   | Batch {batch_idx + 1}/{num_train_batches} - Loss: {batch_train_cost:.4f}", end='\r')

            avg_train_cost = total_training_cost / num_train_batches
            training_costs.append(avg_train_cost)

            if verbose:
                print()

            if validation_set is not None:
                validation_mini_batches = self.random_mini_batch(X_val, Y_val, mini_batch_size, seed)
                total_validation_cost = 0
                num_val_batches = len(validation_mini_batches)

                for batch_idx, mini_batch in enumerate(validation_mini_batches):
                    mini_batch_X, mini_batch_Y = mini_batch
    
                    AL, self.regularize_penalty = self.forward(mini_batch_X, training=False)
                    batch_val_cost = self.compute_cost(mini_batch_Y, AL)
                    total_validation_cost += batch_val_cost

                    if verbose:
                            print(f"Validation | Batch {batch_idx + 1}/{num_val_batches} - Loss: {batch_val_cost:.4f}", end='\r')
                if verbose:
                        print()

                avg_val_cost = total_validation_cost / num_val_batches
                validation_costs.append(avg_val_cost)
            
            if decay:
                self.learning_rate = decay(learning_rate, epoch, decay_rate)
                learning_rates.append(self.learning_rate)

        return training_costs, validation_costs, learning_rates

    def predict(self, X_test):
        AL, _ = self.forward(X_test, training = False)
        return AL




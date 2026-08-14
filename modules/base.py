from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self):
        self.freeze = False
        self.cache = None

    @abstractmethod
    def forward(self, A_prev, training=True):
        pass

    @abstractmethod
    def backward(self, dA):
        pass

    def init_params(self, previous_layer_neurons=None):
        pass

    def init_optimizer(self, optimizer=None):
        pass

    def update_parameters(self, learning_rate=0.01, optimizer=None):
        pass

    def get_regularization_penalty(self, m):
        return 0.0
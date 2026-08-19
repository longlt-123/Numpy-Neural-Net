import numpy as np
import matplotlib.pyplot as plt

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

from neural_network.MLP.mlp_net import MLP


if __name__ == "__main__":
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    Y_train = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    nn = MLP(input_dim=X_train.shape)

    nn.add(Dense(number_neurons=4, activation="linear", init_type="random", regularizer=None, freeze=False))
    nn.add(Activation(activation="sigmoid"))
    nn.add(Dense(number_neurons=1, activation="linear", init_type="random", regularizer=None, freeze=False))
    nn.add(Activation(activation="sigmoid"))

    training_set = (X_train, Y_train)

    training_costs, validation_costs, learning_rates = nn.fit(
        training_set=training_set,
        validation_set=None,
        num_epochs=10000, 
        cost_function="binary_cross_entropy",
        optimizer="adam",
        learning_rate=0.006,
        mini_batch_size=4,
        verbose=True
    )

    predictions = nn.predict(X_train)

    for i in range(len(X_train)):
        prob = predictions[i][0]

        pred_label = 1 if prob >= 0.5 else 0
        true_label = Y_train[i][0]
        
        print(f"Đầu vào X: {X_train[i]} | Dự đoán: {prob:.4f} -> {pred_label} | Thực tế: {true_label}")

    epochs = range(1, len(training_costs) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_costs, label='Training Loss', color='blue', linewidth=2)

    if validation_costs:
        plt.plot(epochs, validation_costs, label='Validation Loss', color='red', linewidth=2, linestyle='--')

    plt.title('Biểu đồ biểu diễn Loss qua từng Epoch', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Binary Cross Entropy)', fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

from modules.base import Layer
from modules.activation_layer import Activation
from modules.dense_layer import Dense
from modules.batchnorm import BatchNorm
from modules.dropout import Dropout
from modules.simple_rnn import Simple_RNN

from neural_network.RNN.rnn import RNN


if __name__ == "__main__":
    X_train = np.array([
        [[1.0], [2.0], [3.0], [4.0]],
        [[2.0], [1.0], [4.0], [3.0]],
        [[3.0], [2.0], [1.0], [2.0]],
        [[1.0], [3.0], [2.0], [1.0]],
        [[4.0], [1.0], [2.0], [3.0]],
    ], dtype=np.float64)

    Y_train = np.array([
        [[1.0], [3.0], [6.0], [10.0]],
        [[2.0], [3.0], [7.0], [10.0]],
        [[3.0], [5.0], [6.0], [8.0]],
        [[1.0], [4.0], [6.0], [7.0]],
        [[4.0], [5.0], [7.0], [10.0]],
    ], dtype=np.float64)

    nn = RNN(input_dim=X_train.shape)

    nn.add(Simple_RNN(hidden_state_dim=8, init_type="he", bidirectional = False, regularizer=None, lambd=0.01, freeze=False))
    nn.add(Dense(number_neurons=1, activation="linear", init_type="random", regularizer=None, freeze=False))
    nn.add(Activation(activation="linear"))

    training_set = (X_train, Y_train)

    training_costs, validation_costs, learning_rates = nn.fit(
        training_set=training_set,
        validation_set=None,
        num_epochs=10000, 
        cost_function="mse",
        optimizer="adam",
        learning_rate=0.006,
        mini_batch_size=4,
        verbose=True
    )

    predictions = nn.predict(X_train)

    for i in range(X_train.shape[0]):
        pred = predictions[i]
        true_label = Y_train[i]
        print(f"Đầu vào X: {X_train[i].flatten()} -> Dự đoán: {pred.flatten()} | Thực tế: {true_label.flatten()}")

    epochs = range(1, len(training_costs) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_costs, label='Training Loss', color='blue', linewidth=2)

    if validation_costs:
        plt.plot(epochs, validation_costs, label='Validation Loss', color='red', linewidth=2, linestyle='--')

    plt.title('Biểu đồ biểu diễn Loss qua từng Epoch', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()
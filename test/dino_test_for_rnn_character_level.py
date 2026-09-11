import numpy as np
import matplotlib.pyplot as plt
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from neural_network.RNN.rnn import RNN
from modules.lstm_rnn import LSTM
from modules.simple_rnn import Simple_RNN
from functions.utilities import prepare_sequence_data
from modules.dense_layer import Dense 
from modules.activation_layer import Activation 

file_path = "data/dinos.txt"
with open(file_path, "r", encoding="utf-8") as file:
    words = [list(line.strip().lower() + '\n') for line in file if line.strip()]

print(f"Tổng số từ: {len(words)}")
print(f"Từ đầu tiên dạng thô (Raw): {repr(words[0])}")

chars = sorted(list(set(char for word in words for char in word)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
print(f"Kích thước tập từ vựng (chars): {len(char_to_idx)}")
vocab_size = len(char_to_idx)
START_IDX = vocab_size
END_IDX = vocab_size + 1
PAD_IDX = vocab_size + 2
UNKNOWN_IDX = vocab_size + 3

vocab_size = vocab_size + 4

X_train_idx, X_train_oh, _, _, _, x_char_to_idx, x_idx_to_char = prepare_sequence_data(
    words, vocab_size, char_to_idx, idx_to_char, max_len=None, 
    shift_mode="left", shift=1, SHIFT_IDX=START_IDX, 
    PAD_IDX=PAD_IDX, START_IDX=START_IDX, END_IDX=END_IDX, UNKNOWN_IDX=UNKNOWN_IDX
)

Y_train_idx, Y_train_oh, Y_train_oh_masked, _, _, y_char_to_idx, y_idx_to_char = prepare_sequence_data(
    words, vocab_size, char_to_idx, idx_to_char, max_len=None, 
    shift_mode="right", shift=1, SHIFT_IDX=END_IDX, 
    PAD_IDX=PAD_IDX, START_IDX=START_IDX, END_IDX=END_IDX, UNKNOWN_IDX=UNKNOWN_IDX
)
num_classes = vocab_size
print(X_train_idx.shape)
print(X_train_oh.shape)
print(X_train_idx[0, :])
print(X_train_oh[0, :, :])

print(Y_train_idx.shape)
print(Y_train_oh.shape)
print(Y_train_idx[0, :])
print(Y_train_oh[0, :, :])

model = RNN(input_dim=X_train_oh.shape, char_to_idx=x_char_to_idx, idx_to_char=x_idx_to_char)
model.add(Simple_RNN(hidden_state_dim=64, bidirectional=False, init_type="he"))
model.add(Dense(number_neurons=num_classes))
model.add(Activation("softmax"))

print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
training_costs, validation_costs, _ = model.fit(
    training_set=(X_train_oh, Y_train_oh_masked),
    num_epochs=70,
    cost_function="categorical_cross_entropy",
    optimizer="adam",
    learning_rate=0.005,
    mini_batch_size=64,
    verbose=True
)

print("\n\n--- TÊN KHỦNG LONG MỚI ĐƯỢC SINH RA ---")
for i in range(10):
    name, _ = model.sampling(seed=i, temperature=1.0, max_length=20)
    
    print(f"{i+1}. {name.capitalize()}")

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
import numpy as np
import matplotlib.pyplot as plt

from neural_network.RNN.rnn import RNN
from modules.simple_rnn import Simple_RNN
from functions.utilities import prepare_sequence_data
from modules.dense_layer import Dense 
from modules.activation_layer import Activation 

file_path = "dinos.txt"
with open(file_path, "r", encoding="utf-8") as file:
    words = [line.strip().lower() + '\n' for line in file if line.strip()]

print(f"Tổng số từ: {len(words)}")
print(f"Từ đầu tiên dạng thô (Raw): {repr(words[0])}")

chars = sorted(list(set(''.join(words))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
print(f"Kích thước tập từ vựng (chars): {len(char_to_idx)}")

X_train, Y_train, mask, max_len = prepare_sequence_data(words, char_to_idx)
num_classes = len(char_to_idx) + 4


model = RNN(input_dim=X_train.shape)
model.add(Simple_RNN(hidden_state_dim=64, bidirectional=False, init_type="he"))
model.add(Dense(number_neurons=num_classes))
model.add(Activation("softmax"))

print("\n--- BẮT ĐẦU HUẤN LUYỆN ---")
training_costs, validation_costs, _ = model.fit(
    training_set=(X_train, Y_train),
    num_epochs=100,
    cost_function="categorical_cross_entropy",
    optimizer="adam",
    learning_rate=0.001,
    mini_batch_size=64,
    verbose=True
)

print("\n\n--- TÊN KHỦNG LONG MỚI ĐƯỢC SINH RA ---")
for i in range(10):
    name = model.sampling(char_to_idx, idx_to_char, seed=i, temperature=1.0, max_length=20)
    
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
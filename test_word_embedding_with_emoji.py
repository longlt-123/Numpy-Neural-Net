import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from neural_network.RNN.rnn import RNN
from modules.lstm_rnn import LSTM
from modules.simple_rnn import Simple_RNN
from functions.utilities import prepare_sequence_data
from functions.score import accuracy_score
from modules.dense_layer import Dense 
from modules.activation_layer import Activation
from modules.embedding import Embedding
from modules.dropout import Dropout

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def word_to_index(X):
    X_indices = np.zeros((X.shape[0], maxLen))
    for i in range(X.shape[0]):
        sentence_words = X[i].lower().split()
        j = 0

        for w in sentence_words:
            if w in word_to_index:
                X_indices[i, j] = word_to_index[w]
                j += 1
    return X_indices

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

df = pd.read_csv("data/train_emoji.csv")
df.drop(df.columns[[2, 3]], axis=1, inplace=True)

X_train = df.iloc[:, 0].to_numpy()
Y_train = df.iloc[:, 1].to_numpy()

print(X_train.shape)
print(X_train)
print(Y_train.shape)
print(Y_train)

maxLen = len(max(X_train, key=lambda x: len(x.split())).split())
print(maxLen)

df_test = pd.read_csv('data/tesss.csv')
X_test = df_test.iloc[:, 0].to_numpy()
Y_test = df_test.iloc[:, 1].to_numpy()

print(X_test.shape)
print(X_test)
print(Y_test.shape)
print(Y_test)

X_train_indices = np.zeros((X_train.shape[0], maxLen))
for i in range(X_train.shape[0]):
    sentence_words = X_train[i].lower().split()
    j = 0

    for w in sentence_words:
        if w in word_to_index:
            X_train_indices[i, j] = word_to_index[w]
            j += 1

print(X_train_indices.shape)
print(X_train_indices)

vocab_size = len(word_to_index) + 1
any_word = next(iter(word_to_vec_map.keys()))
emb_dim = word_to_vec_map[any_word].shape[0]

print(vocab_size)
print(any_word)
print(emb_dim)

emb_matrix = np.zeros((vocab_size, emb_dim))
for word, idx in word_to_index.items():
    emb_matrix[idx, :] = word_to_vec_map[word]

Y_train_oh = convert_to_one_hot(Y_train, C = 5)

model = RNN(input_dim=vocab_size)
model.add(Embedding(emb_dim, input_type="idx", transfer_weights=emb_matrix, freeze=False))
model.add(LSTM(hidden_state_dim=128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(hidden_state_dim=128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(number_neurons=5, activation="linear"))
model.add(Activation("softmax"))

X_train_indices = X_train_indices.astype(np.int32)

training_costs, validation_costs, _ = model.fit(
    training_set=(X_train_indices, Y_train_oh),
    num_epochs=50,
    mini_batch_size=24,
    cost_function="categorical_cross_entropy",
    optimizer="adam",
    learning_rate=0.001,
    verbose=True)

X_test_indices = np.zeros((X_test.shape[0], maxLen))
for i in range(X_test.shape[0]):
    sentence_words = X_test[i].lower().split()
    j = 0

    for w in sentence_words:
        if w in word_to_index:
            X_test_indices[i, j] = word_to_index[w]
            j += 1

X_test_indices = X_test_indices.astype(np.int32)

print(X_test_indices.shape)
print(X_test_indices)

y_pred = model.predict(X_test_indices)
acc = accuracy_score(Y_test, y_pred)
print(acc)

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
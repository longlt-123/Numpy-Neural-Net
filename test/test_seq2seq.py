import numpy as np
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

import matplotlib.pyplot as plt

from neural_network.RNN.rnn import RNN
from neural_network.Seq2Seq.seq2seq import Seq2Seq
from modules.lstm_rnn import LSTM
from modules.simple_rnn import Simple_RNN
from functions.utilities import prepare_sequence_data
from modules.dense_layer import Dense 
from modules.activation_layer import Activation
from modules.embedding import Embedding
from modules.dropout import Dropout
from modules.attention import Attention

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

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(os.path.join(root_dir, "data", 'glove.6B.50d.txt'))

vocab_size = len(word_to_index) + 1
any_word = next(iter(word_to_vec_map.keys()))
emb_dim = word_to_vec_map[any_word].shape[0]

START_IDX = vocab_size
END_IDX = vocab_size + 1
PAD_IDX = 0
UNKNOWN_IDX = vocab_size + 2

print(vocab_size)
print(any_word)
print(emb_dim)

vocab_size = vocab_size + 3  # Adjust for special tokens
emb_matrix = np.zeros((vocab_size, emb_dim))

for word, idx in word_to_index.items():
    emb_matrix[idx, :] = word_to_vec_map[word] 

emb_matrix[START_IDX, :] = np.random.randn(emb_dim) * 0.01
emb_matrix[END_IDX, :] = np.random.randn(emb_dim) * 0.01
emb_matrix[PAD_IDX, :] = np.zeros(emb_dim)
emb_matrix[UNKNOWN_IDX, :] = np.random.randn(emb_dim) * 0.01

X_train = ["hello world", "good morning"]
Y_train = ["bonjour monde", "bon matin"]

X_train, _, _, _, _, x_word_to_idx, x_idx_to_word = prepare_sequence_data(X_train, vocab_size, word_to_index, index_to_word, max_len=None, shift_mode=None, shift=1, SHIFT_IDX=START_IDX, PAD_IDX=PAD_IDX, START_IDX=START_IDX, END_IDX=END_IDX, UNKNOWN_IDX=UNKNOWN_IDX)
Y_train_input, _, _, _, _, y_word_to_idx, y_idx_to_word = prepare_sequence_data(Y_train, vocab_size, word_to_index, index_to_word, max_len=None, shift_mode="left", shift=1, SHIFT_IDX=START_IDX, PAD_IDX=PAD_IDX, START_IDX=START_IDX, END_IDX=END_IDX, UNKNOWN_IDX=UNKNOWN_IDX)
Y_train_target, Y_train_target_oh, Y_train_target_oh_masked, _, _, y_word_to_idx, y_idx_to_word = prepare_sequence_data(Y_train, vocab_size, word_to_index, index_to_word, max_len=None, shift_mode="right", shift=1, SHIFT_IDX=END_IDX, PAD_IDX=PAD_IDX, START_IDX=START_IDX, END_IDX=END_IDX, UNKNOWN_IDX=UNKNOWN_IDX)
terminal_word = ["<START>", "<END>", "<PAD>", "<UNK>"]
print("Vocabulary size:", len(x_word_to_idx))
print("Input sequence shape:", X_train.shape)
print("Input sequence:", X_train)

print(Y_train_input)
print(Y_train_target)
print(Y_train_target_oh)
print(Y_train_target_oh_masked)

encoder = RNN(input_dim=vocab_size)
encoder.add(Embedding(embedding_dim=emb_dim, transfer_weights=emb_matrix, freeze=True))
encoder.add(LSTM(hidden_state_dim=64, bidirectional=True, init_type="he", return_sequences=True, merge_mode="concat"))

decoder_emb = Embedding(
    embedding_dim=emb_dim, 
    transfer_weights=emb_matrix,
    freeze=False
)
decoder_lstm = LSTM(hidden_state_dim=64*2, return_sequences=True)

decoder = RNN(
    input_dim=vocab_size, 
    layers=[decoder_emb, decoder_lstm], 
    char_to_idx=y_word_to_idx.copy(), 
    idx_to_char=y_idx_to_word.copy()
)

attention_layer = Attention(mode="dense", context_mode="concat")
attention_layer.init_params(s_dim=64*2, h_dim=64*2)
decoder_lstm.init_params(input_dims=emb_dim + 64*2)

model = Seq2Seq(encoder, decoder)
model.add(attention_layer)
model.add(Dense(vocab_size, activation="softmax"))

print("Test luồng forward và backward có Attention")
training_costs, validation_costs, _ = model.fit(
    training_set=(X_train, Y_train_input, Y_train_target_oh_masked), 
    num_epochs=50, 
    cost_function="categorical_cross_entropy",
    learning_rate=0.005,
    optimizer="adam", 
    mini_batch_size=2
)

print("Test Attention thành công! Loss:", training_costs[-1])

X_test = np.array(["hello world", "good morning"])
X_test, _, _, _, _, x_word_to_idx, x_idx_to_word = prepare_sequence_data(X_test, vocab_size, word_to_index, index_to_word, max_len=None, shift_mode=None, shift=1, SHIFT_IDX=START_IDX, PAD_IDX=PAD_IDX, START_IDX=START_IDX, END_IDX=END_IDX, UNKNOWN_IDX=UNKNOWN_IDX)
sampled_sequences = model.sampling(X_test, temperature=1.0, max_length=20)
print ("Sampled sequences:", sampled_sequences)
for i in range(len(sampled_sequences)):
    seq = sampled_sequences[i]
    word_seq = [y_idx_to_word[idx] + " " for idx in seq]
    print(f"Input: {X_test[i]} | Sampled Output: {''.join(word_seq)}")

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
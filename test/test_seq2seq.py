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
        
        i = 0
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(os.path.join(root_dir, "data", 'glove.6B.50d.txt'))

vocab_size = len(word_to_index)
any_word = next(iter(word_to_vec_map.keys()))
emb_dim = word_to_vec_map[any_word].shape[0]

print(vocab_size)
print(any_word)
print(emb_dim)

emb_matrix = np.zeros((vocab_size, emb_dim))

for word, idx in word_to_index.items():
    emb_matrix[idx, :] = word_to_vec_map[word] 

X_train = np.array(["hello world", "good morning"])
Y_train = np.array(["bonjour monde", "bon matin"])

X_train, _, _, _, _, _, x_word_to_idx, x_idx_to_word = prepare_sequence_data(X_train, word_to_index, index_to_word, max_len=5+1)
X_train = X_train[:, 1:]  # Remove the START token for input to the model
terminal_word = ["<START>", "<END>", "<PAD>", "<UNK>"]
print("Vocabulary size:", len(x_word_to_idx))
print("Input sequence shape:", X_train.shape)
print("Input sequence:", X_train)

encoder = RNN(input_dim=vocab_size)
encoder.add(Embedding(embedding_dim=emb_dim, transfer_weights=emb_matrix, terminal_word=terminal_word, freeze=True))
encoder.add(LSTM(hidden_state_dim=64, bidirectional=True, init_type="he", return_sequences=True, merge_mode="concat"))

decoder_emb = Embedding(
    embedding_dim=emb_dim, 
    transfer_weights=emb_matrix, 
    terminal_word=terminal_word, 
    freeze=True
)
decoder_lstm = LSTM(hidden_state_dim=64*2, return_sequences=True)

decoder = RNN(
    input_dim=vocab_size, 
    layers=[decoder_emb, decoder_lstm], 
    char_to_idx=word_to_index.copy(), 
    idx_to_char=index_to_word.copy()
)

attention_layer = Attention(mode="dense", context_mode="concat")
attention_layer.init_params(s_dim=64*2, h_dim=64*2)
decoder_lstm.init_params(input_dims=emb_dim + 64*2)

model = Seq2Seq(encoder, decoder)
model.add(attention_layer)
model.add(Dense(vocab_size + 4, activation="softmax"))

print("Test luồng forward và backward có Attention")
training_costs, _, _ = model.fit(
    training_set=(X_train, Y_train), 
    num_epochs=5, 
    cost_function="categorical_cross_entropy", 
    optimizer="adam", 
    mini_batch_size=2
)

print("Test Attention thành công! Loss:", training_costs[-1])
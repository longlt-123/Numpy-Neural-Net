import numpy as np
import math
from functions.activations import linear, relu, sigmoid, leaky_relu
from functions.output import softmax

def shuffle_data(X, Y):
    m = X.shape[0]
    permutation = np.random.permutation(m)
    shuffle_X = X[permutation, :]
    if Y is not None:
        shuffle_Y = Y[permutation, :]
    else:
        shuffle_Y = None

    return shuffle_X, shuffle_Y

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

def prepare_sequence_data(words, char_to_idx, idx_to_char, max_len=None):
    """
    Chuyển đổi danh sách các từ thành ma trận đầu vào X, Y dạng one-hot kèm mask.
    """
    vocab_size = len(char_to_idx)
    
    START_IDX = vocab_size
    END_IDX = vocab_size + 1
    PAD_IDX = vocab_size + 2
    UNKNOWN_IDX = vocab_size + 3

    char_to_idx["<START>"] = START_IDX
    char_to_idx["<END>"] = END_IDX
    char_to_idx["<PAD>"] = PAD_IDX
    char_to_idx["<UNK>"] = UNKNOWN_IDX

    idx_to_char[START_IDX] = "<START>"
    idx_to_char[END_IDX] = "<END>"
    idx_to_char[PAD_IDX] = "<PAD>"
    idx_to_char[UNKNOWN_IDX] = "<UNK>"

    num_classes = vocab_size + 4

    if max_len is None:
        max_len = max(len(w) for w in words) + 1
        
    m = len(words)
    
    X_idx = np.full((m, max_len), PAD_IDX, dtype=int)
    Y_idx = np.full((m, max_len), PAD_IDX, dtype=int)
    mask = np.zeros((m, max_len), dtype=int)
    
    for i, word in enumerate(words):
        chars = [char_to_idx[c] for c in word if c in char_to_idx]
        unknown_chars = [UNKNOWN_IDX for c in word if c not in char_to_idx]
        if unknown_chars:
            chars += unknown_chars

        x_seq = [START_IDX] + chars
        y_seq = chars + [END_IDX]
        
        x_seq = x_seq[:max_len]
        y_seq = y_seq[:max_len]
        
        length = len(x_seq)
        X_idx[i, :length] = x_seq
        Y_idx[i, :length] = y_seq

        mask[i, :length] = 1
    X_onehot = np.zeros((m, max_len, num_classes))
    Y_onehot = np.zeros((m, max_len, num_classes))
    mask_3d = mask[:, :, np.newaxis]
    
    for i in range(m):
        for t in range(max_len):
            X_onehot[i, t, X_idx[i, t]] = 1
            Y_onehot[i, t, Y_idx[i, t]] = 1

    Y_onehot = Y_onehot * mask_3d

    return X_idx, Y_idx, X_onehot, Y_onehot, mask, max_len, char_to_idx, idx_to_char

def convert_targets(targets: np.ndarray, to: str = None, threshold = 0.5):
    if to is None:
        if targets.ndim == 1:
            num_classes = len(set(targets))
            converted_targets = np.eye(num_classes)[targets] #One-hot encoding
        else:
            converted_targets = targets
    
    elif to == "categorical":
        num_classes = len(set(targets))
        converted_targets = np.eye(num_classes)[targets]
    
    elif to == "one_hot":
        idx = np.argmax(targets, axis=-1)
        converted_targets = np.zeros(targets.shape)
        converted_targets[np.arange(targets.shape[0]), idx] = 1
    
    elif to == "binary":
        converted_targets = np.where(targets >= threshold, 1, 0)
    
    elif to == "labels":
        if targets.ndim != 1:
            converted_targets = np.argmax(targets, axis=-1)
        else:
            converted_targets = targets
    
    elif to == "probability":
        converted_targets = softmax(targets)
    
    else:
        raise ValueError(f"Unsupported target format: {to}")

    return converted_targets

def initialize_parameters(input_size, output_size, init_type="he"):
    np.random.seed(1)

    if init_type == "zero":
        parameter = np.zeros((input_size, output_size))
    elif init_type == "one":
        parameter = np.ones((input_size, output_size))
    elif init_type == "random":
        parameter = np.random.randn(input_size, output_size) * 0.01
    elif init_type == "he":
        parameter = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
    else:
        raise ValueError(f"Unsupported initialize type: {init_type}")

    return parameter

def initialize_optimizer(W, b, optimizer="adam"):
    v = {}
    s = {}

    if optimizer == "gd":
        return {}, {}

    elif optimizer == "rmsprop":
        s["dW"] = np.zeros_like(W)
        s["db"] = np.zeros_like(b)
        return {}, s

    elif optimizer == "momentum":
        v["dW"] = np.zeros_like(W)
        v["db"] = np.zeros_like(b)
        return v, {}

    elif optimizer == "adam":
        v["dW"] = np.zeros_like(W)
        v["db"] = np.zeros_like(b)

        s["dW"] = np.zeros_like(W)
        s["db"] = np.zeros_like(b)

        return v, s

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

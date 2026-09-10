import numpy as np
import math
from functions.activations import linear, relu, sigmoid, leaky_relu
from functions.output import softmax

def random_mini_batch(X, Y1=None, Y2=None, mini_batch_size = 64, seed = 0):
        if seed != 0:
            np.random.seed(seed)
        
        m = X.shape[0]
        permutation = np.random.permutation(m)
        shuffle_X = X[permutation, :]
        shuffle_Y1 = Y1[permutation, :] if Y1 is not None else None
        shuffle_Y2 = Y2[permutation, :] if Y2 is not None else None

        mini_batches = []

        num_complete_minibatches = m // mini_batch_size
        for k in range(num_complete_minibatches):
            mini_batch_X = shuffle_X[k * mini_batch_size : (k+1) * mini_batch_size, :]
            if shuffle_Y1 is not None:
                mini_batch_Y1 = shuffle_Y1[k * mini_batch_size : (k+1) * mini_batch_size, :]
            else:
                mini_batch_Y1 = None
            if shuffle_Y2 is not None:
                mini_batch_Y2 = shuffle_Y2[k * mini_batch_size : (k+1) * mini_batch_size, :]
            else:
                mini_batch_Y2 = None
            mini_batch = (mini_batch_X, mini_batch_Y1, mini_batch_Y2)
            mini_batches.append(mini_batch)

        if m % mini_batch_size != 0:
            mini_batch_X = shuffle_X[num_complete_minibatches * mini_batch_size :, :]
            if shuffle_Y1 is not None:
                mini_batch_Y1 = shuffle_Y1[num_complete_minibatches * mini_batch_size :, :]
            else:
                mini_batch_Y1 = None
            if shuffle_Y2 is not None:
                mini_batch_Y2 = shuffle_Y2[num_complete_minibatches * mini_batch_size :, :]
            else:
                mini_batch_Y2 = None
            mini_batch = (mini_batch_X, mini_batch_Y1, mini_batch_Y2)
            mini_batches.append(mini_batch)

        return mini_batches

def convert_sequence_to_indices(sequence, word_to_idx, UNKNOWN_IDX):
    return [word_to_idx.get(word, UNKNOWN_IDX) for word in sequence]

def pad_sequence(sequence, maxlen, padding='post', truncating='post', PAD_IDX=0):
    if len(sequence) == 0:
        raise ValueError("Input sequence is empty.")
        
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]

    padded_sequence = np.full(maxlen, PAD_IDX, dtype=int)
    mask = np.zeros(maxlen, dtype=int)

    if padding == 'post':
        padded_sequence[:len(trunc)] = trunc
        mask[:len(trunc)] = 1
    else:
        padded_sequence[-len(trunc):] = trunc
        mask[-len(trunc):] = 1

    return padded_sequence, mask

def shift_sequence(sequence, mode="left", shift=1, SHIFT_IDX=0):
    shifted = np.full_like(sequence, SHIFT_IDX)
    if shift < len(sequence):
        if mode == "left":
            shifted[shift:] = sequence[:-shift]
        elif mode == "right":
            shifted[:-shift] = sequence[shift:]
        else:
            raise ValueError("Mode must be 'left' or 'right'")
    return shifted

def sequences_one_hot_encode(sequences, num_classes):
    one_hot = np.zeros((len(sequences), len(sequences[0]), num_classes))
    for i, seq in enumerate(sequences):
        for j, idx in enumerate(seq):
            if idx < num_classes:
                one_hot[i, j, idx] = 1
    return one_hot

def prepare_sequence_data(sentences, vocab_size, word_to_idx, idx_to_word, max_len=None, shift_mode=None, shift=1, SHIFT_IDX=0, PAD_IDX=0, START_IDX=None, END_IDX=None, UNKNOWN_IDX=None):
    tmp_word_to_idx = word_to_idx.copy()
    tmp_idx_to_word = idx_to_word.copy()

    tmp_word_to_idx["<START>"] = START_IDX
    tmp_word_to_idx["<END>"] = END_IDX
    tmp_word_to_idx["<PAD>"] = PAD_IDX
    tmp_word_to_idx["<UNK>"] = UNKNOWN_IDX

    tmp_idx_to_word[START_IDX] = "<START>"
    tmp_idx_to_word[END_IDX] = "<END>"
    tmp_idx_to_word[PAD_IDX] = "<PAD>"
    tmp_idx_to_word[UNKNOWN_IDX] = "<UNK>"

    num_classes = vocab_size
    tokenized_sentences = []
    for sentence in sentences:
        if isinstance(sentence, (str, np.str_)):
            tokenized_sentences.append(sentence.strip().split())
        else:
            tokenized_sentences.append(sentence)

    if max_len is None:
        if shift_mode is not None:
            max_len = max(len(s) for s in tokenized_sentences) + shift
        else:
            max_len = max(len(s) for s in tokenized_sentences)
    m = len(tokenized_sentences)

    mask = np.zeros((m, max_len), dtype=int)
    seq_idx = np.zeros((m, max_len), dtype=int)
    
    for i, tokens in enumerate(tokenized_sentences):
        seq_indices = convert_sequence_to_indices(tokens, tmp_word_to_idx, UNKNOWN_IDX)
        seq_padded, seq_mask = pad_sequence(seq_indices, maxlen=max_len, padding='post', truncating='post', PAD_IDX=PAD_IDX)
        mask[i, :] = seq_mask
        if shift_mode is not None:
            seq_shifted = shift_sequence(seq_padded, mode=shift_mode, shift=shift, SHIFT_IDX=SHIFT_IDX)
        else:
            seq_shifted = seq_padded
        seq_idx[i, :] = seq_shifted

    mask_3d = mask[:, :, np.newaxis]
    seq_oh = sequences_one_hot_encode(seq_idx, num_classes)
    seq_oh_masked = seq_oh * mask_3d

    return seq_idx, seq_oh, seq_oh_masked, mask, max_len, tmp_word_to_idx, tmp_idx_to_word

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

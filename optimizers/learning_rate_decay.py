import numpy as np

def decay_with_decay_rate(learning_rate0, epoch_num, decay_rate):
    return learning_rate0 / (1 + decay_rate * epoch_num)

def decay_with_stage(learning_rate0, epoch_num, num_epoch_per_stage=1000):
    return learning_rate0 / (1 + np.floor(epoch_num / num_epoch_per_stage))
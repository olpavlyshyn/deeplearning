def exponential_decay(learning_rate, epoch_num, decay_rate):
    return 1 / (1 + decay_rate * epoch_num) * learning_rate

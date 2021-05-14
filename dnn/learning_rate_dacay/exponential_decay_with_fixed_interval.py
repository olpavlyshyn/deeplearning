def exponential_decay_with_fixed_interval(learning_rate, epoch_num, decay_rate, time_interval):
    if epoch_num % time_interval == 0:
        learning_rate = 1 / (1 + decay_rate * (epoch_num / time_interval)) * learning_rate
    return learning_rate

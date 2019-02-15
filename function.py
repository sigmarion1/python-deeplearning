import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def step_function(x):
    y = x > 0
    return y.astype(np.int)

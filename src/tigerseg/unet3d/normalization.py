import numpy as np


def max_normalize(data):
    return data / np.max(data)


def min_max_normalize(data):
    min = np.min(data)
    return (data - min) / (np.max(data) - min)
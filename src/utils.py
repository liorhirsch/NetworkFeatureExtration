import numpy as np


def pad_layer_outputs(layer_statistics, max_layer_size):
    pad_size = max_layer_size - len(layer_statistics)
    return np.pad(layer_statistics, (0, pad_size))

import numpy as np


def pad_with_columns(layer_statistics, max_layer_size):
    pad_size = max_layer_size - len(layer_statistics)
    return np.pad(layer_statistics, (0, pad_size))

def pad_with_rows(f_map, rows_target):
    return np.pad(f_map, ((0, rows_target - f_map.shape[0]), (0, 0)))

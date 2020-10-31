import numpy as np


def mad(data: list):
    data = np.array(data)
    median = np.median(data)
    deviations = data - median
    absolute_deviations = np.abs(deviations)
    mad = np.median(absolute_deviations)

    return mad


def compute_reconstruction_error(input_values, reconstruction):
    deviations = input_values - reconstruction
    reconstruction_error = np.sum(np.abs(deviations), axis=1)

    return reconstruction_error

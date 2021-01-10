import numpy as np


def mad(data):
    data = np.array(data)
    median = np.median(data)
    deviations = data - median
    absolute_deviations = np.abs(deviations)
    mad = np.median(absolute_deviations)

    return mad


def compute_reconstruction_error(input_values, reconstruction, has_multiple_features: bool = True):
    deviations = input_values - reconstruction
    reconstruction_error = np.abs(deviations)

    if has_multiple_features:
        reconstruction_error = np.sum(reconstruction_error, axis=1)
    else:
        reconstruction_error = reconstruction_error.flatten()

    return reconstruction_error

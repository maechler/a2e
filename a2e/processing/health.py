import math
import numpy as np
from typing import Callable
from a2e.processing.stats import mad


def health_score_sigmoid_activation(x: float, shrink: float = 0.75, shift: float = 5.0, x_max: float = 100):
    # to keep very large numbers from generating errors
    if x > x_max:
        x = x_max

    return -(math.exp(shrink * x - shift) / (math.exp(shrink * x - shift) + 1)) + 1


def compute_health_score(train_error: list, test_error: list, mode: str = 'median', health_score_activation: Callable = health_score_sigmoid_activation) -> list:
    if mode is 'median':
        train_center = np.median(train_error)
        train_deviation = mad(train_error)
    elif mode is 'average':
        train_center = np.average(train_error)
        train_deviation = np.std(train_error)
    else:
        raise ValueError(f'Invalid mode "{mode}" provided.')

    health_score = []

    for current_error in test_error:
        if math.isnan(current_error):
            continue

        magnitude_of_deviation = abs(current_error - train_center) / train_deviation
        health = health_score_activation(magnitude_of_deviation)

        health_score.append(health)

    return health_score

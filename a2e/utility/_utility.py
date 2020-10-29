import numpy as np
from datetime import datetime, timezone, tzinfo
from itertools import product
from typing import Callable


def timestamp_to_date_time(timestamp: float, tz: tzinfo = timezone.utc) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=tz)


def grid_run(param_grid: dict, run_callable: Callable):
    param_grid_keys = list(param_grid)
    param_grid_values = param_grid.values()

    for param_comb in product(*param_grid_values):
        param_dict = {}

        for i, param_value in enumerate(param_comb):
            param_name = param_grid_keys[i]
            param_dict[param_name] = param_value

        run_callable(param_dict)


def build_samples(data: np.ndarray, target_sample_length: int, target_dimensions: int = 2):
    if len(data.shape) == 1:
        drop_data = len(data) % target_sample_length
        samples = data[:-drop_data]
    else:
        input_sample_length = data.shape[1]
        drop_data = input_sample_length % target_sample_length
        drop_start = input_sample_length - drop_data
        samples = np.delete(data, np.s_[drop_start:input_sample_length], axis=1)

    return samples.reshape(-1, target_sample_length) if target_dimensions == 2 else samples.reshape(-1, target_sample_length, 1)

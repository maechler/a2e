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

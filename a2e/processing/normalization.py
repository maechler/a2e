import numpy as np
from typing import Union, Tuple
from sklearn.preprocessing import MinMaxScaler


def min_max_scale(
    data: np.ndarray,
    feature_range: Tuple[float, float] = (0, 1),
    return_scaler: bool = False,
    fit_mode: str = 'per_feature',
) -> Union[Tuple[np.ndarray, MinMaxScaler], np.ndarray]:
    scaler = MinMaxScaler(feature_range=feature_range)

    if fit_mode == 'per_feature':
        scaler.fit(data)
        scaled_data = scaler.transform(data)
    elif fit_mode == 'per_sample':
        scaled_data = data

        for i in range(0, len(scaled_data)):
            scaled_data[i] = scaler.fit_transform(scaled_data[i].reshape(-1, 1)).reshape(1, -1)
    else:
        raise ValueError(f'Invalid fit_mode "{fit_mode}" provided.')

    return (scaled_data, scaler) if return_scaler else scaled_data

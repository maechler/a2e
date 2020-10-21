import numpy as np
from typing import Union, Tuple
from sklearn.preprocessing import MinMaxScaler


def min_max_scale(data: np.ndarray, feature_range: Tuple[float, float] = (0, 1), return_scaler: bool = False) -> Union[Tuple[np.ndarray, MinMaxScaler], np.ndarray]:
    scaler = MinMaxScaler(feature_range=feature_range)

    scaler.fit(data)
    scaled_data = scaler.transform(data)

    return (scaled_data, scaler) if return_scaler else scaled_data

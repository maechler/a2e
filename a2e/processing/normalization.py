import numpy as np
from typing import Callable
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class VoidScaler:
    def fit(self, data):
        return self

    def transform(self, data):
        return data

    def fit_transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class Scaler:
    def __init__(self, scaler: Callable, fit_mode: str = 'per_feature', **scaler_kwargs):
        self.fit_mode = fit_mode
        self.scaler = scaler(**scaler_kwargs)
        self.n_features = 0

        if fit_mode not in ['per_feature', 'per_sample']:
            raise ValueError(f'Invalid fit_mode "{fit_mode}" provided.')

    def fit(self, data):
        if self.fit_mode == 'per_feature':
            self.scaler = self.scaler.fit(data)
        else:
            self.n_features = len(data)
            self.scaler = self.scaler.fit(data.T)

    def transform(self, data):
        if self.fit_mode == 'per_feature':
            return self.scaler.transform(data)
        else:
            if len(data) < self.n_features:
                fill_up_size = self.n_features - len(data)
                fill_up = np.ones((fill_up_size, data.shape[1]))
                filled_up_data = np.concatenate((data, fill_up))

                transformed = self.scaler.transform(filled_up_data.T).T

                return transformed[:-fill_up_size, :]
            else:
                return self.scaler.transform(data.T).T

    def fit_transform(self, data):
        self.fit(data)

        return self.transform(data)

    def inverse_transform(self, data):
        if self.fit_mode == 'per_feature':
            return self.scaler.inverse_transform(data)
        else:
            if len(data) < self.n_features:
                fill_up_size = self.n_features - len(data)
                fill_up = np.ones((fill_up_size, data.shape[1]))
                filled_up_data = np.concatenate((data, fill_up))

                transformed = self.scaler.inverse_transform(filled_up_data.T).T

                return transformed[:-fill_up_size, :]
            else:
                return self.scaler.inverse_transform(data.T).T


def create_scaler(scaler: str) -> Scaler:
    if scaler in ['none', 'void']:
        return Scaler(VoidScaler)
    if scaler == 'min_max_scale':
        return Scaler(MinMaxScaler, fit_mode='per_feature')
    elif scaler == 'min_max_scale_per_sample':
        return Scaler(MinMaxScaler, fit_mode='per_sample')
    elif scaler == 'std_scale':
        return Scaler(StandardScaler, fit_mode='per_feature')
    elif scaler == 'std_scale_per_sample':
        return Scaler(StandardScaler, fit_mode='per_sample')
    else:
        raise ValueError(f'Invalid scaler "{scaler}" provided.')

import gzip
import os
import yaml
import pathlib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Callable, Union, List, Dict
from tensorflow.keras.utils import get_file
from pandas import DataFrame, Series
from a2e.utility import timestamp_to_date_time


class BearingDataSet:
    window_cache = {}

    def __init__(self, data_set_key: str, data_frame: DataFrame, windows: dict):
        self.data_set_key = data_set_key
        self.data_frame = data_frame
        self.windows = windows

    def all(self, column, as_numpy=False, modifier: Callable = None, drop_duplicates: bool = True, add_label: bool = False) -> Union[DataFrame, np.ndarray]:
        windowed_data_frames = []

        for window_key, window_description in self.windows.items():
            windowed_data_frames.append(self.windowed_data(column, window=window_key, as_numpy=False, modifier=modifier, drop_duplicates=drop_duplicates, add_label=add_label))

        windowed_data_frame = pd.concat(windowed_data_frames)

        return windowed_data_frame.to_numpy() if as_numpy else windowed_data_frame

    def train(self, column, as_numpy=False, modifier: Callable = None, drop_duplicates: bool = True, add_label: bool = False) -> Union[DataFrame, np.ndarray]:
        return self.windowed_data(column, window='train', as_numpy=as_numpy, modifier=modifier, drop_duplicates=drop_duplicates, add_label=add_label)

    def test(self, column, split=None, as_numpy=False, modifier: Callable = None, drop_duplicates: bool = True, add_label: bool = False) -> Union[DataFrame, List[DataFrame], np.ndarray, List[np.ndarray]]:
        windowed_data_frames = {}

        for window_key, window_description in self.windows.items():
            if not window_key.startswith('test_'):
                continue

            windowed_data_frames[window_key] = self.windowed_data(column, window=window_key, as_numpy=False, modifier=modifier, drop_duplicates=drop_duplicates, add_label=add_label)

        if split is not None:
            if split == '2fold':
                healthy_windows = []
                anomalous_windows = []

                for window_key, window_data_frame in windowed_data_frames.items():
                    if 'healthy' in window_key:
                        healthy_windows.append(window_data_frame)
                    else:
                        anomalous_windows.append(window_data_frame)

                healthy_data_frame = pd.concat(healthy_windows)
                anomalous_data_frame = pd.concat(anomalous_windows)

                if as_numpy:
                    return [healthy_data_frame.to_numpy(), anomalous_data_frame.to_numpy()]
                else:
                    return [healthy_data_frame, anomalous_data_frame]
            elif split == 'nfold':
                return list(map(lambda x: x.to_numpy(), windowed_data_frames.values())) if as_numpy else windowed_data_frames.values()
            else:
                raise ValueError(f'Invalid split option "{split}" provided.')
        else:
            windowed_data_frame = pd.concat(windowed_data_frames.values())

            return windowed_data_frame.to_numpy() if as_numpy else windowed_data_frame

    def as_dict(self, column, as_numpy=False, modifier: Callable = None, split_test=False, drop_duplicates: bool = True, add_label: bool = False, labels_only: bool = False) -> Union[Dict[str, DataFrame], Dict[str, Series]]:
        add_label = add_label or labels_only
        datasets_dict = {
            'all': self.all(column=column, as_numpy=as_numpy, modifier=modifier, drop_duplicates=drop_duplicates, add_label=add_label),
            'train': self.train(column=column, as_numpy=as_numpy, modifier=modifier, drop_duplicates=drop_duplicates, add_label=add_label),
        }

        if split_test:
            for window_key, window_description in self.windows.items():
                if window_key.startswith('test_'):
                    datasets_dict[window_key] = self.windowed_data(column=column, window=window_key, as_numpy=as_numpy, modifier=modifier, drop_duplicates=drop_duplicates, add_label=add_label)
        else:
            datasets_dict['test'] = self.test(column=column, split=None, as_numpy=as_numpy, modifier=modifier, drop_duplicates=drop_duplicates, add_label=add_label)

        if labels_only:
            for key, data_frame in datasets_dict.items():
                datasets_dict[key] = data_frame['label']

        return datasets_dict

    def windowed_data(self, column: str, window: str = None, as_numpy=False, modifier: Callable = None, drop_duplicates: bool = True, add_label: bool = False) -> Union[DataFrame, np.ndarray]:
        cache_hash = str(hash(f'{self.data_set_key}{column}{window}{as_numpy}{modifier}{drop_duplicates}{add_label}'))

        if cache_hash not in self.window_cache:
            if window is not None:
                windowed_data_frame = self.data_frame.loc[self.windows[window]['mask']].copy()
            else:
                windowed_data_frame = self.data_frame.copy()

            if add_label:
                windowed_data_frame['label'] = self.windows[window]['label']

            if drop_duplicates:
                if column == 'fft':
                    self.drop_adjacent_duplicates(windowed_data_frame, ['fft_1', 'fft_2', 'fft_3'])
                else:
                    self.drop_adjacent_duplicates(windowed_data_frame, [column])

            if modifier is not None:
                windowed_data_frame = modifier(windowed_data_frame)

            if column == 'fft':
                windowed_data_frame = windowed_data_frame.iloc[:, 4:]
            else:
                windowed_data_frame = windowed_data_frame[[column]]

            self.window_cache[cache_hash] = windowed_data_frame.to_numpy() if as_numpy else windowed_data_frame

        return self.window_cache[cache_hash].copy()

    def drop_adjacent_duplicates(self, data_frame: DataFrame, columns: list):
        previous_row = None
        indexes_to_drop = []

        for index, row in data_frame.iterrows():
            if previous_row is not None:
                if previous_row[columns].equals(row[columns]):
                    indexes_to_drop.append(index)

            previous_row = row

        data_frame.drop(index=indexes_to_drop, inplace=True)


def load_data(data_set_key: str, a2e_data_path: str = '../../../a2e-data/data', cache_dir: str = None) -> BearingDataSet:
    """Loads one of the bearing datasets.

    Parameters
    ----------
    data_set_key: str
        One of the available dataset keys `400rpm`, `800rpm`, `1200rpm`, `variable_rpm`

    a2e_data_path: str
        Local file path to the a2e-data repository

    cache_dir: str
        Optional cache directory for the datasets, defaults to `~/.a2e/` or `/tmp/.a2e/` as a fallback

    Returns
    -------
    data_frame, windows: DataFrame, dict
        A data_frame indexed by timestamp and a dictionary containing data set windows for `train`, `test`, `test_healthy` and `test_anomalous`
    """

    if a2e_data_path is not None and not a2e_data_path.startswith('http') and not a2e_data_path.startswith('file://'):
        if os.path.isabs(a2e_data_path):
            a2e_data_path = 'file://' + os.path.abspath(a2e_data_path)
        else:
            bearing_module_path = pathlib.Path(__file__).parent.absolute()
            absolute_data_path = os.path.abspath(os.path.join(bearing_module_path, a2e_data_path))
            if os.name == 'nt':
                absolute_data_path = f'/{absolute_data_path}'.replace('\\', '/')

            a2e_data_path = 'file://' + absolute_data_path

    if not os.path.isdir(a2e_data_path):
        a2e_data_path = 'https://github.com/maechler/a2e-data/raw/master/data/'

    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), '.a2e')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    a2e_data_path = a2e_data_path.rstrip('/') + '/'
    data_set_description_origin = f'{a2e_data_path}{data_set_key}.yaml'
    data_set_origin = f'{a2e_data_path}{data_set_key}.csv.gz'
    data_set_description_path = get_file(data_set_key + '.yaml', origin=data_set_description_origin, cache_dir=cache_dir, cache_subdir='datasets/bearing')
    windows = {}

    with open(data_set_description_path) as data_set_description_file:
        data_set_description = yaml.load(data_set_description_file, Loader=yaml.FullLoader)
        data_set_path = get_file(data_set_key + '.csv.gz', origin=data_set_origin, cache_dir=cache_dir, cache_subdir='datasets/bearing', file_hash=data_set_description['data']['md5_hash'], hash_algorithm='md5')

    with gzip.open(data_set_path, mode='rt') as data_set_file:
        data_frame = pd.read_csv(data_set_file, parse_dates=[data_set_description['data']['index_column']], date_parser=lambda x: timestamp_to_date_time(float(x)), quotechar='"', sep=',')
        data_frame = data_frame.set_index(data_set_description['data']['index_column'])

    for window_key, window_description in data_set_description['windows'].items():
        windows[window_key] = {
            'mask': (data_frame.index > window_description['start']) & (data_frame.index <= window_description['end']),
            'label': window_description['label'],
        }

    return BearingDataSet(data_set_key, data_frame, windows)

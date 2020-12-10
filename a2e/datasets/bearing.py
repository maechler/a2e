import gzip
import os
import yaml
import pathlib
import pandas as pd
from pathlib import Path
from typing import Callable, Union, Tuple
from tensorflow.keras.utils import get_file
from pandas import DataFrame
from a2e.utility import timestamp_to_date_time


class BearingDataSet:

    def __init__(self, data_frame: DataFrame, masks: dict):
        self.data_frame = data_frame
        self.masks = masks

    def all(self, column, as_numpy=False, modifier: Callable = None, drop_duplicates: bool = True) -> DataFrame:
        return self.masked_data(column, as_numpy=as_numpy, modifier=modifier, drop_duplicates=drop_duplicates)

    def train(self, column, as_numpy=False, modifier: Callable = None) -> DataFrame:
        return self.masked_data(column, mask='train', as_numpy=as_numpy, modifier=modifier)

    def test(self, column, split=False, as_numpy=False, modifier: Callable = None) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        if split:
            test_healthy = self.masked_data(column, mask='test_healthy', as_numpy=as_numpy, modifier=modifier)
            test_anomalous = self.masked_data(column, mask='test_anomalous', as_numpy=as_numpy, modifier=modifier)

            return test_healthy, test_anomalous
        else:
            return self.masked_data(column, mask='test', as_numpy=as_numpy, modifier=modifier)

    def as_dict(self, column, as_numpy=False, modifier: Callable = None, split_test=False):
        datasets_dict = {
            'train': self.train(column=column, as_numpy=as_numpy, modifier=modifier),
            'all': self.all(column=column, as_numpy=as_numpy, modifier=modifier)
        }

        if split_test:
            test_healthy, test_anomalous = self.test(column=column, as_numpy=as_numpy, modifier=modifier, split=True)
            datasets_dict['test_healthy'] = test_healthy
            datasets_dict['test_anomalous'] = test_anomalous
        else:
            datasets_dict['test'] = self.test(column=column, as_numpy=as_numpy, modifier=modifier)

        return datasets_dict

    def masked_data(self, column, mask: str = None, as_numpy=False, modifier: Callable = None, drop_duplicates: bool = True) -> DataFrame:
        if mask is not None:
            masked_data_frame = self.data_frame.loc[self.masks[mask]]
        else:
            masked_data_frame = self.data_frame

        if column == 'fft':
            masked_data_frame = masked_data_frame.iloc[:, 4:]
        else:
            masked_data_frame = masked_data_frame[[column]]

        if drop_duplicates:
            if column == 'fft':
                self.drop_adjacent_duplicates(masked_data_frame, ['fft_1', 'fft_2', 'fft_3'])
            else:
                self.drop_adjacent_duplicates(masked_data_frame, [column])

        if modifier is not None:
            masked_data_frame = modifier(masked_data_frame)

        return masked_data_frame.to_numpy() if as_numpy else masked_data_frame

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
    data_frame, masks: DataFrame, dict
        A data_frame indexed by timestamp and a dictionary containing data set masks for `train`, `test`, `test_healthy` and `test_anomalous`
    """
    if a2e_data_path is None:
        a2e_data_path = 'https://github.com/maechler/a2e-data/raw/master/data/'

    if not a2e_data_path.startswith('http') and not a2e_data_path.startswith('file://'):
        if os.path.isabs(a2e_data_path):
            a2e_data_path = 'file://' + os.path.abspath(a2e_data_path)
        else:
            bearing_module_path = pathlib.Path(__file__).parent.absolute()
            absolute_data_path = os.path.abspath(os.path.join(bearing_module_path, a2e_data_path))
            if os.name == 'nt':
                absolute_data_path = f'/{absolute_data_path}'.replace('\\', '/')

            a2e_data_path = 'file://' + absolute_data_path

    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), '.a2e')

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    a2e_data_path = a2e_data_path.rstrip('/') + '/'
    data_set_description_origin = f'{a2e_data_path}{data_set_key}.yaml'
    data_set_origin = f'{a2e_data_path}{data_set_key}.csv.gz'
    data_set_description_path = get_file(data_set_key + '.yaml', origin=data_set_description_origin, cache_dir=cache_dir, cache_subdir='datasets/bearing')
    data_set_description = yaml.load(open(data_set_description_path), Loader=yaml.FullLoader)
    data_set_path = get_file(data_set_key + '.csv.gz', origin=data_set_origin, cache_dir=cache_dir, cache_subdir='datasets/bearing', file_hash=data_set_description['data']['md5_hash'], hash_algorithm='md5')

    data_frame = pd.read_csv(gzip.open(data_set_path, mode='rt'), parse_dates=[data_set_description['data']['index_column']], date_parser=lambda x: timestamp_to_date_time(float(x)), quotechar='"', sep=',')
    data_frame = data_frame.set_index(data_set_description['data']['index_column'])
    masks = {
        'train': (data_frame.index > data_set_description['windows']['train']['start']) & (data_frame.index <= data_set_description['windows']['train']['end']),
        'test': (data_frame.index > data_set_description['windows']['test_healthy']['start']) & (data_frame.index <= data_set_description['windows']['test_anomalous']['end']),
        'test_healthy': (data_frame.index > data_set_description['windows']['test_healthy']['start']) & (data_frame.index <= data_set_description['windows']['test_healthy']['end']),
        'test_anomalous': (data_frame.index > data_set_description['windows']['test_anomalous']['start']) & (data_frame.index <= data_set_description['windows']['test_anomalous']['end']),
    }

    return BearingDataSet(data_frame, masks)

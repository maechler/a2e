import gzip
import os
import yaml
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

    def all(self, column, as_numpy=False, modifier: Callable = None) -> DataFrame:
        return self.masked_data(column, as_numpy=as_numpy, modifier=modifier)

    def train(self, column, as_numpy=False, modifier: Callable = None) -> DataFrame:
        return self.masked_data(column, mask='train', as_numpy=as_numpy, modifier=modifier)

    def test(self, column, split=False, as_numpy=False, modifier: Callable = None) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        if split:
            test_healthy = self.masked_data(column, mask='test_healthy', as_numpy=as_numpy, modifier=modifier)
            test_anomalous = self.masked_data(column, mask='test_anomalous', as_numpy=as_numpy, modifier=modifier)

            return test_healthy, test_anomalous
        else:
            return self.masked_data(column, mask='test', as_numpy=as_numpy, modifier=modifier)

    def as_dict(self, column, as_numpy=False, modifier: Callable = None):
        return {
            'train': self.train(column=column, as_numpy=as_numpy, modifier=modifier),
            'test':  self.test(column=column, as_numpy=as_numpy, modifier=modifier),
            'all': self.all(column=column, as_numpy=as_numpy, modifier=modifier)
        }

    def masked_data(self, column, mask: str = None, as_numpy=False, modifier: Callable = None) -> DataFrame:
        if mask is not None:
            masked_data_frame = self.data_frame.loc[self.masks[mask]]
        else:
            masked_data_frame = self.data_frame

        if modifier is not None:
            masked_data_frame = modifier(masked_data_frame)

        if column == 'fft':
            masked_data_frame = masked_data_frame.iloc[:, 4:]
        else:
            masked_data_frame = masked_data_frame[[column]]

        return masked_data_frame.to_numpy() if as_numpy else masked_data_frame


def load_data(data_set_key: str, a2e_data_path: str = '../../a2e_data/data', cache_dir: str = None) -> BearingDataSet:
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
    else:
        a2e_data_path = 'file://' + os.path.abspath(a2e_data_path)

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

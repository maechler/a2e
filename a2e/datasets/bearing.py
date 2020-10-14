import gzip
import os
from pathlib import Path
import yaml
from keras.utils import get_file
import pandas as pd
from pandas import DataFrame
from a2e.utility import timestamp_to_date_time


def load_data(data_set_key: str, a2e_data_path: str = '../../a2e_data/data', cache_dir: str = None) -> (DataFrame, dict):
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

    data_frame = pd.read_csv(gzip.open(data_set_path, mode='rt'), parse_dates=[data_set_description['data']['index_column']], date_parser=timestamp_to_date_time, quotechar='"', sep=',')
    data_frame = data_frame.set_index(data_set_description['data']['index_column'])
    masks = {
        'train': (data_frame.index > data_set_description['windows']['train']['start']) & (data_frame.index <= data_set_description['windows']['train']['end']),
        'test': (data_frame.index > data_set_description['windows']['test_healthy']['start']) & (data_frame.index <= data_set_description['windows']['test_anomalous']['end']),
        'test_healthy': (data_frame.index > data_set_description['windows']['test_healthy']['start']) & (data_frame.index <= data_set_description['windows']['test_healthy']['end']),
        'test_anomalous': (data_frame.index > data_set_description['windows']['test_anomalous']['start']) & (data_frame.index <= data_set_description['windows']['test_anomalous']['end']),
    }

    return data_frame, masks


def mask_data_frame(data_frame: DataFrame, mask: dict) -> DataFrame:
    data_frame_mask = (data_frame.index > mask['start']) & (data_frame.index <= mask['end'])

    return data_frame.loc[data_frame_mask]

import pandas as pd
import yaml
import os
from a2e.input import AbstractInput, DataSet
from a2e.pipeline import PipelineData
from a2e.utility import timestamp_to_date_time


class TestDataInput(AbstractInput):

    def __init__(self, config):
        super().__init__(config)

        self.data_config = None
        self.data_path = None
        self.index_column = None

    def init(self):
        self.data_config = yaml.load(open(self.get_config('data_config_path')), Loader=yaml.FullLoader)
        self.index_column = self.data_config['data']['index_column']

        if os.path.isabs(self.data_config['data']['path']):
            self.data_path = self.data_config['data']['path']
        else:
            data_config_dir = os.path.dirname(self.get_config('data_config_path'))
            self.data_path = os.path.abspath(os.path.join(data_config_dir, '..', self.data_config['data']['path']))

    def process(self, pipeline_data: PipelineData):
        data_frame = pd.read_csv(self.data_path, parse_dates=[self.index_column], date_parser=timestamp_to_date_time, quotechar='"', sep=',')
        data_frame = data_frame.set_index(self.index_column)
        masks = {
            'train': {
                'start': self.data_config['windows']['train']['start'],
                'end': self.data_config['windows']['train']['end'],
            },
            'test': {
                'start': self.data_config['windows']['test_healthy']['start'],
                'end': self.data_config['windows']['test_anomalous']['end'],
            },
            'test_healthy': {
                'start': self.data_config['windows']['test_healthy']['start'],
                'end': self.data_config['windows']['test_healthy']['end'],
            },
            'test_anomalous': {
                'start': self.data_config['windows']['test_anomalous']['start'],
                'end': self.data_config['windows']['test_anomalous']['end'],
            },
        }

        data_set = DataSet(data_frame, masks)
        pipeline_data.data_set = data_set

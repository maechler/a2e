from abc import ABC
from pandas import DataFrame
from a2e.pipeline import AbstractPipelineStep


class AbstractInput(AbstractPipelineStep, ABC):
    """Base class for input steps"""

    def __init__(self, config):
        super().__init__(config)

        #self.index_column = self.get_config('index_column')


class DataSet:

    def __init__(self, data_frame: DataFrame, masks: dict = {}):
        self._data_frame = data_frame
        self.masks = masks

    @property
    def all_data(self) -> DataFrame:
        return self._data_frame

    def masked_data(self, mask: str) -> DataFrame:
        if mask in self.masks:
            data_frame_mask = self.mask(mask)

            return self._data_frame.loc[data_frame_mask]
        else:
            raise ValueError(f'Mask "{mask}" is not supported.')

    def mask(self, mask: str):
        return (self._data_frame.index > self.masks[mask]['start']) & (self._data_frame.index <= self.masks[mask]['end'])

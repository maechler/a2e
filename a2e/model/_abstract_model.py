from abc import ABC, abstractmethod
from functools import partial

from a2e.evaluation import EvaluationResult
from a2e.processing.normalization import min_max_scale, std_scale


class AbstractModel(ABC):
    @abstractmethod
    def evaluate(self, config, x_train, y_train, x_valid, y_valid, budget=None, **kwargs) -> EvaluationResult:
        pass

    def pre_process(self, config, *args):
        if '_pre_processing' in config:
            pre_processing = config['_pre_processing']

            if pre_processing == 'min_max_scale':
                return list(map(min_max_scale, args))
            elif pre_processing == 'min_max_scale_per_sample':
                return list(map(partial(min_max_scale, fit_mode='per_sample'), args))
            elif pre_processing == 'std_scale':
                return list(map(std_scale, args))
            elif pre_processing == 'std_scale_per_sample':
                return list(map(partial(std_scale, fit_mode='per_sample'), args))

        return args

    def load_config(self, config: dict = None, **kwargs):
        raise NotImplementedError

    def fit(self, config, x_train, y_train, x_valid=None, y_valid=None, budget=None, **kwargs):
        raise NotImplementedError

    def predict(self, x, **kwargs):
        raise NotImplementedError

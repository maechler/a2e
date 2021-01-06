from abc import ABC, abstractmethod
from a2e.evaluation import EvaluationResult


class AbstractModel(ABC):
    @abstractmethod
    def evaluate(self, config, x_train, y_train, x_valid, y_valid, budget=None, **kwargs) -> EvaluationResult:
        pass

    def pre_process(self, config, *args):
        if '_pre_processing' in config:
            pre_processing_function = config['_pre_processing']

            return list(map(pre_processing_function, args))
        else:
            return args

    def load_config(self, config: dict = None, **kwargs):
        raise NotImplementedError

    def fit(self, config, x_train, y_train, x_valid=None, y_valid=None, budget=None, **kwargs):
        raise NotImplementedError

    def predict(self, x, **kwargs):
        raise NotImplementedError

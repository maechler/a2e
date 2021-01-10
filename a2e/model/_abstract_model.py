from abc import ABC, abstractmethod
from a2e.evaluation import EvaluationResult


class AbstractModel(ABC):
    @abstractmethod
    def evaluate(self, x_train, y_train, x_valid, y_valid, budget=None, **kwargs) -> EvaluationResult:
        pass

    @abstractmethod
    def load_config(self, config: dict = None, **kwargs):
        pass

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, budget=None, **kwargs):
        raise NotImplementedError

    def predict(self, x, **kwargs):
        raise NotImplementedError

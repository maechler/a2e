import multiprocessing
import secrets
import ConfigSpace
import a2e
from pandas import DataFrame
from typing import Optional
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from a2e.evaluation import EvaluationResult
from a2e.model import AbstractModel
from a2e.optimizer import OptimizationResult
from a2e.utility import synchronized


class AbstractOptimizer(ABC):
    def __init__(
        self,
        configuration_space: ConfigSpace,
        model: AbstractModel,
        x,
        y=None,
        max_iterations: int = 50,
        min_budget: Optional[int] = None,
        max_budget: Optional[int] = None,
        eta: int = 2,
        validation_split: float = 0.1,
        validation_split_shuffle: bool = True,
        run_id: str = None,
    ):
        self.configuration_space = configuration_space
        self.model = model
        self.max_iterations = max_iterations
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.run_id = run_id if run_id is not None else secrets.token_hex(nbytes=16)
        self.evaluation_result_aggregator = EvaluationResultAggregator()

        if validation_split is not None:
            self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(
                x,
                x if y is None else y,
                test_size=validation_split,
                shuffle=validation_split_shuffle
            )
        else:
            self.x_train = self.x_valid = x
            self.y_train = self.y_valid = x if y is None else y

    @abstractmethod
    def optimize(self) -> OptimizationResult:
        pass

    def best_configuration(self) -> dict:
        return self.configuration_by_percentile_rank(1.0)

    def configuration_by_percentile_rank(self, percentile_rank: float = 1.0) -> dict:
        optimization_result = OptimizationResult(self.evaluation_result_aggregator.get_evaluation_results())

        return optimization_result.configuration_by_percentile_rank(percentile_rank)

    def best_model(self) -> AbstractModel:
        self.model.load_config(self.best_configuration())

        return self.model

    def refit_by_percentile_rank(self, percentile_rank: float = 1.0) -> AbstractModel:
        configuration = self.configuration_by_percentile_rank(percentile_rank)

        self.model.fit(configuration, self.x_train, self.y_train, self.x_valid, self.y_valid, self.max_budget)

        return self.model

    def refit_best_model(self) -> AbstractModel:
        return self.refit_by_percentile_rank(1.0)


class EvaluationResultAggregator:
    def __init__(self):
        process_manager = multiprocessing.Manager()

        self._lock = process_manager.RLock()
        self._state = process_manager.dict()
        self._state['evaluation_id'] = 0
        self._state['best_evaluation_id'] = 0
        self._state['best_cost'] = float('inf')
        self._state['evaluation_results'] = process_manager.list()

    @synchronized
    def add_evaluation_result(self, evaluation_result: EvaluationResult):
        evaluation_result.set_id(self.get_new_evaluation_id())

        if evaluation_result.cost < self._state['best_cost']:
            self._state['best_evaluation_id'] = evaluation_result.id
            self._state['best_cost'] = evaluation_result.cost

        evaluation_result.add_info('best_evaluation_id', self._state['best_evaluation_id'])

        self._state['evaluation_results'].append(evaluation_result)

    @synchronized
    def get_new_evaluation_id(self):
        self._state['evaluation_id'] = self._state['evaluation_id'] + 1

        return self._state['evaluation_id']

    @synchronized
    def get_evaluation_results(self, as_data_frame=False):
        if as_data_frame:
            return DataFrame(list(map(lambda x: x.to_dict(), list(self._state['evaluation_results']))))
        else:
            return list(self._state['evaluation_results'])


def create_optimizer(optimizer_type: str, *args, **kwargs) -> AbstractOptimizer:
    # Using a2e.optimizer.* to avoid circular dependency
    if optimizer_type == 'BayesianGaussianProcessOptimizer':
        return a2e.optimizer.BayesianGaussianProcessOptimizer(*args, **kwargs)
    elif optimizer_type == 'BayesianRandomForrestOptimizer':
        return a2e.optimizer.BayesianRandomForrestOptimizer(*args, **kwargs)
    elif optimizer_type == 'BayesianHyperbandOptimizer':
        return a2e.optimizer.BayesianHyperbandOptimizer(*args, **kwargs)
    elif optimizer_type == 'HyperbandOptimizer':
        return a2e.optimizer.HyperbandOptimizer(*args, **kwargs)
    elif optimizer_type == 'RandomOptimizer':
        return a2e.optimizer.RandomOptimizer(*args, **kwargs)
    else:
        raise ValueError(f'Invalid optimizer type "{optimizer_type}" provided.')

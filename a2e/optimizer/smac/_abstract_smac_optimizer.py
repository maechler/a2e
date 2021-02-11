import os
import shutil
import ConfigSpace
import tempfile
from abc import abstractmethod
from typing import Optional, Dict, Callable
from smac.scenario.scenario import Scenario
from a2e.optimizer import AbstractOptimizer, OptimizationResult
from a2e.optimizer.smac import ModelTaeRunner
from a2e.model import AbstractModel


class AbstractSMACOptimizer(AbstractOptimizer):
    def __init__(
        self,
        config_space: ConfigSpace,
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
        scenario_kwargs: Optional[Dict] = None,
        runner_kwargs: Optional[Dict] = None,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        super().__init__(config_space, model, x, y, max_iterations, min_budget, max_budget, eta, validation_split, validation_split_shuffle, run_id)

        scenario_kwargs = scenario_kwargs if scenario_kwargs is not None else {}
        runner_kwargs = runner_kwargs if runner_kwargs is not None else {}
        optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        self.temp_dir = os.path.join(tempfile.gettempdir(), 'smac')
        self.scenario = Scenario({
            'run_obj': 'quality',
            'runcount-limit': self.max_iterations,
            'cs': self.config_space,
            'deterministic': 'true',
            'output_dir': self.temp_dir,
            'abort_on_first_run_crash': False,
            **scenario_kwargs,
        })
        self.runner = ModelTaeRunner(**({
            'model': self.model,
            'evaluation_result_aggregator': self.evaluation_result_aggregator,
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_valid': self.x_valid,
            'y_valid': self.y_valid,
            **runner_kwargs,
        }))

        optimizer_arguments = {
            'scenario': self.scenario,
            'tae_runner': self.runner,
        }

        if self.min_budget is not None and self.max_budget is not None:
            optimizer_arguments['intensifier_kwargs'] = {
                'initial_budget': self.min_budget,
                'max_budget': self.max_budget,
                'eta': self.eta
            }

        self.optimizer = self.get_optimizer_class()(**({
            **optimizer_arguments,
            **optimizer_kwargs,
        }))

    @abstractmethod
    def get_optimizer_class(self) -> Callable:
        pass

    def optimize(self) -> OptimizationResult:
        try:
            self.optimizer.optimize()
        finally:
            shutil.rmtree(self.temp_dir)

        return OptimizationResult(
            evaluation_results=self.evaluation_result_aggregator.get_evaluation_results(),
        )

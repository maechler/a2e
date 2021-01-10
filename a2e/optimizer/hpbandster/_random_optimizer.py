from typing import Callable
from hpbandster.core.master import Master
from hpbandster.optimizers import RandomSearch
from a2e.optimizer.hpbandster import AbstractHpbandsterOptimizer


class RandomOptimizer(AbstractHpbandsterOptimizer):
    def create_optimizer(self) -> Master:
        self.min_budget = self.max_budget

        return self.get_optimizer_class()(**({
            'configspace': self.config_space,
            'run_id': self.run_id,
            'min_budget': self.min_budget,
            'max_budget': self.max_budget,
            'eta': self.eta,
            **self.optimizer_kwargs,
        }))

    def get_optimizer_class(self) -> Callable:
        return RandomSearch

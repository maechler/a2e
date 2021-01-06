from typing import Callable
from hpbandster.optimizers import BOHB
from a2e.optimizer.hpbandster import AbstractHpbandsterOptimizer


class BayesianHyperbandOptimizer(AbstractHpbandsterOptimizer):
    def get_optimizer_class(self) -> Callable:
        return BOHB

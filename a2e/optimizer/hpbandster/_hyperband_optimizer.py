from typing import Callable
from hpbandster.optimizers import HyperBand
from a2e.optimizer.hpbandster import AbstractHpbandsterOptimizer


class HyperbandOptimizer(AbstractHpbandsterOptimizer):
    def get_optimizer_class(self) -> Callable:
        return HyperBand

from typing import Callable
from smac.facade.smac_bohb_facade import BOHB4HPO
from a2e.optimizer.smac._abstract_smac_optimizer import AbstractSMACOptimizer


class BayesianRandomForrestOptimizer(AbstractSMACOptimizer):
    def get_optimizer_class(self) -> Callable:
        return BOHB4HPO

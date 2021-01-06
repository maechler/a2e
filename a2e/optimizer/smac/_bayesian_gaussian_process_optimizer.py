from typing import Callable
from smac.facade.smac_bo_facade import SMAC4BO
from a2e.optimizer.smac._abstract_smac_optimizer import AbstractSMACOptimizer


class BayesianGaussianProcessOptimizer(AbstractSMACOptimizer):
    def get_optimizer_class(self) -> Callable:
        return SMAC4BO

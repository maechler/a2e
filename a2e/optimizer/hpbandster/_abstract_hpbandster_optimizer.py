from abc import abstractmethod
from typing import Optional, Dict, Callable
import ConfigSpace
import hpbandster.core.nameserver as hpns
from hpbandster.core.master import Master
from a2e.model import AbstractModel
from a2e.optimizer import OptimizationResult, AbstractOptimizer
from a2e.optimizer.hpbandster import ModelWorker


class AbstractHpbandsterOptimizer(AbstractOptimizer):
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
        nameserver_ip: str = '127.0.0.1',
        nameserver_port: Optional[int] = None,
        worker_kwargs: Optional[Dict] = None,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        super().__init__(configuration_space, model, x, y, max_iterations, min_budget, max_budget, eta, validation_split, validation_split_shuffle, run_id)

        self.worker_kwargs = worker_kwargs if worker_kwargs is not None else {}
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        self.nameserver_ip = nameserver_ip
        self.nameserver_port = nameserver_port
        self.nameserver = hpns.NameServer(run_id=self.run_id, host=self.nameserver_ip, port=self.nameserver_port)
        self.nameserver.start()

        self.worker = ModelWorker(**({
            'model': self.model,
            'evaluation_result_aggregator': self.evaluation_result_aggregator,
            'x_train': self.x_train,
            'y_train': self.y_train,
            'x_valid': self.x_valid,
            'y_valid': self.y_valid,
            'run_id': self.run_id,
            'nameserver': self.nameserver_ip,
            'nameserver_port': self.nameserver_port,
            **self.worker_kwargs,
        }))

        self.worker.run(background=True)
        self.optimizer = self.create_optimizer()

    def create_optimizer(self) -> Master:
        return self.get_optimizer_class()(**({
            'configspace': self.configuration_space,
            'run_id': self.run_id,
            'min_budget': self.min_budget,
            'max_budget': self.max_budget,
            'eta': self.eta,
            **self.optimizer_kwargs,
        }))

    @abstractmethod
    def get_optimizer_class(self) -> Callable:
        pass

    def optimize(self) -> OptimizationResult:
        try:
            result = self.optimizer.run(n_iterations=self.max_iterations)
        finally:
            self.optimizer.shutdown(shutdown_workers=True)
            self.nameserver.shutdown()

        return OptimizationResult(self.evaluation_result_aggregator.get_evaluation_results())

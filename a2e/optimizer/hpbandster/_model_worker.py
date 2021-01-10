from hpbandster.core.worker import Worker
from a2e.model import AbstractModel
from a2e.optimizer import EvaluationResultAggregator


class ModelWorker(Worker):
    def __init__(
        self,
        model: AbstractModel,
        evaluation_result_aggregator: EvaluationResultAggregator,
        x_train,
        y_train,
        x_valid,
        y_valid,
        run_id,
        nameserver=None,
        nameserver_port=None,
        logger=None,
        host=None,
        id=None,
        timeout=None,
    ):
        super().__init__(run_id, nameserver=nameserver, nameserver_port=nameserver_port, logger=logger, host=host, id=id, timeout=timeout)

        self.model = model
        self.evaluation_result_aggregator = evaluation_result_aggregator
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def compute(self, config, budget, working_directory, **kwargs):
        iteration, stage, actual_num_config = kwargs['config_id']
        self.model.load_config(config, budget=budget, **kwargs)
        evaluation_result = self.model.evaluate(
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            budget,
        )

        evaluation_result.add_info('iteration', iteration)
        evaluation_result.add_info('stage', stage)
        evaluation_result.add_info('actual_num_config', actual_num_config)

        self.evaluation_result_aggregator.add_evaluation_result(evaluation_result)

        return {
            'loss': evaluation_result.cost,
            'info': evaluation_result.info,
        }

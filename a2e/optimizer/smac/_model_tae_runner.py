from a2e.optimizer._abstract_optimizer import EvaluationResultAggregator
from a2e.model import AbstractModel


class ModelTaeRunner:
    def __init__(
        self,
        model: AbstractModel,
        evaluation_result_aggregator: EvaluationResultAggregator,
        x_train,
        y_train,
        x_valid,
        y_valid,
    ):
        self.model = model
        self.evaluation_result_aggregator = evaluation_result_aggregator
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def __call__(self, config, budget, *args, **kwargs):
        self.model.load_config(dict(config), budget=budget, **kwargs)
        evaluation_result = self.model.evaluate(
            self.x_train,
            self.y_train,
            self.x_valid,
            self.y_valid,
            budget,
        )

        self.evaluation_result_aggregator.add_evaluation_result(evaluation_result)

        return evaluation_result.cost

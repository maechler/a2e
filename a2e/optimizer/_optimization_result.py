from typing import List
from pandas import DataFrame
from a2e.evaluation import EvaluationResult


class OptimizationResult:
    def __init__(self, evaluation_results: List[EvaluationResult]):
        self.evaluation_results_by_id = {}
        self._evaluation_results = evaluation_results
        self.evaluation_results = DataFrame(list(map(lambda x: x.to_dict(), evaluation_results)))

        for evaluation_result in evaluation_results:
            self.evaluation_results_by_id[evaluation_result.id] = evaluation_result

    def best_config(self) -> dict:
        return self.config_by_percentile_rank(1.0)

    def get_evaluation_result_by_id(self, evaluation_result_id) -> EvaluationResult:
        return self.evaluation_results_by_id[evaluation_result_id]

    def config_by_percentile_rank(self, percentile_rank: float = 1.0) -> dict:
        if percentile_rank < 0.0 or percentile_rank > 1.0:
            raise ValueError('Parameter percentile_rank must be within 0.0 and 1.0')

        num_evaluations = len(self.evaluation_results)

        if num_evaluations == 0:
            raise ValueError('No evaluation results found.')

        sorted_results = self.evaluation_results.sort_values(by=['cost'], ascending=False)
        target_index = round(percentile_rank * (num_evaluations - 1))
        target_id = sorted_results.iloc[target_index]['id']

        return self.get_evaluation_result_by_id(target_id).config

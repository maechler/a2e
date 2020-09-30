import math
from a2e.pipeline import PipelineData, AbstractPipelineStep


def health_score_sigmoid(x):
    return -(math.exp(.75 * x - 5) / (math.exp(.75 * x - 5) + 1)) + 1


class HealthScoreStep(AbstractPipelineStep):

    def __init__(self, config: dict):
        super().__init__(config)

        self.source_column = self.get_config('source_column')
        self.target_column = self.get_config('target_column')

    def process(self, pipeline_data: PipelineData):
        health_score = []
        rolling_error_std = pipeline_data.data_frame[self.source_column].std()
        rolling_error_median = pipeline_data.data_frame[self.source_column].median()
        #rolling_error_std = data_frame['reconstruction_error_rolling'].loc[train_mask].std()
        #rolling_error_median = data_frame['reconstruction_error_rolling'].loc[train_mask].median()

        for index, row in pipeline_data.data_frame.iterrows():
            current_error = row[self.source_column]
            magnitude_of_std = abs(current_error - rolling_error_median) / rolling_error_std
            health = health_score_sigmoid(magnitude_of_std)
            health_score.append(health)

        pipeline_data.data_frame[self.target_column] = health_score


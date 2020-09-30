from a2e.algorithm import AbstractAlgorithm
from a2e.pipeline import PipelineData


class VoidAlgorithm(AbstractAlgorithm):
    """Void algorithm, used as placeholder when no algorithm is configured."""
    def process(self, pipeline_data: PipelineData):
        pass

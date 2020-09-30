from a2e.input import AbstractInput
from a2e.pipeline import PipelineData


class VoidInput(AbstractInput):

    def process(self, pipeline_data: PipelineData):
        pass

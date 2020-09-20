from a2e.output import AbstractOutput
from a2e.pipeline import PipelineData


class VoidOutput(AbstractOutput):

    def process(self, pipeline_data: PipelineData):
        pass

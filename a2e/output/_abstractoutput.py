from abc import ABC
from a2e.pipeline import AbstractPipelineStep


class AbstractOutput(AbstractPipelineStep, ABC):
    """Base class for output steps"""

    def __init__(self, config):
        super().__init__(config)

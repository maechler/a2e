import os
from abc import abstractmethod, ABC
import git
from a2e.utility import instance_from_config, get_property_recursively, out_path


class Pipeline:
    def __init__(self, config):
        self.config = config
        keras_backend = self.get_config('keras_backend', default='')

        if keras_backend:
            os.environ['KERAS_BACKEND'] = keras_backend

        self.input = instance_from_config(self.get_config('input', default={'class': 'a2e.input.VoidInput'}))
        self.algorithm = instance_from_config(self.get_config('algorithm', default={'class': 'a2e.algorithm.VoidAlgorithm'}))
        self.output = instance_from_config(self.get_config('output', default={'class': 'a2e.output.VoidOutput'}))
        self.steps = []

        self.steps.append(self.input)
        self.write_log()

        for preprocessing_step in self.get_config('preprocessing', default=[]):
            self.steps.append(instance_from_config(preprocessing_step))

        self.steps.append(self.algorithm)

        for postprocessing_step in self.get_config('postprocessing', default=[]):
            self.steps.append(instance_from_config(postprocessing_step))

        self.steps.append(self.output)

    def run(self) -> None:
        for step in self.steps:
            step.init()

        if self.get_config('mode', default='offline') == 'offline':
            pipeline_data = PipelineData(self)

            try:
                for step in self.steps:
                    step.process(pipeline_data)
            except Exception as e:
                print(e)

            finally:
                for step in self.steps:
                    step.destroy()
        else:
            raise ValueError(f'Mode "{self.get_config("mode")}" is not supported by this pipeline.')

    def get_config(self, *args, **kwargs):
        return get_property_recursively(self.config, *args, **kwargs)

    def write_log(self):
        out_file = out_path('pipeline.log')

        with open(out_file, 'w') as file:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha

            file.write(f'Git commit hash: {sha}')


class PipelineData:
    def __init__(self, pipeline: Pipeline):
        from a2e.input import DataSet  # Delay import because of circular dependency

        self.pipeline: Pipeline = pipeline
        self.data_set: DataSet = None
        self.algorithm_input: any = None


class AbstractPipelineStep(ABC):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def init(self) -> None:
        pass

    def destroy(self) -> None:
        pass

    @abstractmethod
    def process(self, pipeline_data: PipelineData):
        pass

    def get_config(self, *args, **kwargs):
        return get_property_recursively(self.config, *args, **kwargs)

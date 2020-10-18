import inspect
import json
import os
import pathlib

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from a2e.experiment import git_hash, git_diff


class Experiment:

    def __init__(self, experiment_id: str, verbose: bool = True, out_directory: str = None):
        self.experiment_id = experiment_id
        self.out_directory = out_directory
        self.verbose = verbose

        if self.out_directory is None:
            experiment_directory = pathlib.Path(__file__).parent.absolute()
            self.out_directory = os.path.abspath(os.path.join(experiment_directory, f'../../out/{self.experiment_id}/'))

        if not os.path.isdir(self.out_directory):
            os.makedirs(self.out_directory)

        self.log('git/hash', git_hash())
        self.log('git/diff', git_diff())

    def __del__(self):
        print(f'Destructor called for Experiment with id "{self.experiment_id}"')

    def log(self, key: str, value: any):
        out_file_path = self._out_file(key)

        with open(out_file_path, 'w') as out_file:
            if isinstance(value, dict):
                out_file.write(json.dumps(value, indent=2))
            else:
                out_file.write(str(value))

    def log_model(self, model: Model):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))

        plot_model(model, show_shapes=True, expand_nested=True, to_file=self._out_file('keras/model.png'))
        self.log('keras/summary.txt', "\n".join(model_summary))

    def _out_file(self, relative_file_path: str):
        out_file_path = os.path.join(self.out_directory, relative_file_path)
        out_directory = pathlib.Path(out_file_path).parent.absolute()

        if not os.path.isdir(out_directory):
            os.makedirs(out_directory)

        return out_file_path

    def print(self, message: str):
        if self.verbose:
            print(f'[{self.experiment_id}] {message}')


def create_experiment(experiment_id: str = None) -> Experiment:
    if experiment_id is None:
        caller_frame = inspect.stack()[1]
        caller_filename = caller_frame.filename

        experiment_id = os.path.splitext(os.path.basename(caller_filename))[0]

    return Experiment(experiment_id)

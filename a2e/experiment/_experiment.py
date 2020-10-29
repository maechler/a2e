import inspect
import json
import os
import pathlib
import datetime
import shutil
import traceback
from typing import Callable
from numpy.random import seed
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, History
from a2e.experiment import git_hash, git_diff
from a2e.plotter import plot, plot_model_layer_weights
from a2e.utility import grid_run


class Experiment:

    def __init__(self, experiment_id: str = None, verbose: bool = True, out_directory: str = None, auto_datetime_directory: bool = True, set_seeds: bool = True):
        self.experiment_id = experiment_id
        self.experiment_start = datetime.datetime.now()
        self.out_directory = out_directory
        self.verbose = verbose
        self.run_count = 0
        self.run_id = None

        if set_seeds:
            self._set_seeds()

        if self.experiment_id is None:
            caller_frame = inspect.stack()[1]
            caller_filename = caller_frame.filename

            self.experiment_id = os.path.splitext(os.path.basename(caller_filename))[0]

        if self.out_directory is None:
            experiment_directory = pathlib.Path(__file__).parent.absolute()
            self.out_directory = os.path.abspath(os.path.join(experiment_directory, f'../../out/{self.experiment_id}/'))

        if auto_datetime_directory:
            self.out_directory = os.path.join(self.out_directory, self.experiment_start.strftime('%Y_%m_%d/%H_%M_%S'))

        if os.path.isdir(self.out_directory):
            shutil.rmtree(self.out_directory)

        if not os.path.isdir(self.out_directory):
            os.makedirs(self.out_directory)

        self.log('git/hash', git_hash())
        self.log('git/diff', git_diff())

    def __del__(self):
        self.print(f'Destructor called for Experiment with id "{self.experiment_id}"')

    def _set_seeds(self):
        seed(1)
        set_seed(1)

    def log(self, key: str, value: any):
        out_file_path = self._out_path(key)

        with open(out_file_path, 'w') as out_file:
            if isinstance(value, dict) or isinstance(value, list):
                out_file.write(json.dumps(value, indent=2))
            else:
                out_file.write(str(value))

    def log_plot(self, key: str, x=None, y=None, xlabel=None, ylabel=None, ylim=None, label=None, time_formatting: bool = False, create_figure: bool = True, close: bool = True):
        self.log(key, y)
        plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, ylim=ylim, label=label, time_formatting=time_formatting, create_figure=create_figure, close=close, out_path=self._out_path(key))

    def log_history(self, history: History):
        if 'loss' in history.history and 'val_loss' in history.history:
            self.log('metrics/val_loss', history.history['val_loss'])
            self.log_plot('metrics/loss', y=history.history['val_loss'], label='validation loss', xlabel='epoch', close=False)
            self.log_plot('metrics/loss', y=history.history['loss'], label='loss', xlabel='epoch', create_figure=False)
        else:
            if 'loss' in history.history:
                self.log_plot('metrics/loss', y=history.history['loss'], xlabel='epoch', ylabel='loss')

            if 'val_loss' in history.history:
                self.log_plot('metrics/val_loss', y=history.history['val_loss'], xlabel='epoch', ylabel='validation loss')

    def log_model(self, model: Model):
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))

        plot_model(model, show_shapes=True, expand_nested=True, to_file=self._out_path('model/model.png'))
        self.log('model/summary', '\n'.join(model_summary))

        if len(model.get_weights()) > 0:
            plot_model_layer_weights(model, out_path=self._out_path('model/layers', is_directory=True))

    def _out_path(self, relative_path: str, is_directory: bool = False) -> str:
        out_file_path = pathlib.Path(os.path.join(self.out_directory, relative_path))
        out_directory = out_file_path.absolute() if is_directory else out_file_path.parent.absolute()

        if not os.path.isdir(out_directory):
            os.makedirs(out_directory)

        return str(out_file_path)

    def print(self, message: str):
        if self.verbose:
            print(f'[{self.experiment_id}] {message}')

    def callbacks(self, save_best: bool = True):
        callbacks = [
            ModelCheckpoint(self._out_path('model/model.hdf5')),
            CSVLogger(self._out_path('metrics/loss_log.csv'), separator=',', append=False),
        ]

        if save_best:
            callbacks.append(ModelCheckpoint(
                self._out_path('model/model.best.hdf5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min')
            )

        return callbacks

    def grid_run(self, param_grid: dict, run_callable: Callable, auto_run_id: bool = True):
        experiment = self

        def run_callable_wrapper(params):
            if auto_run_id:
                self.run_id = '_'.join(params.values())

            experiment.start_run()
            experiment.log('run_config', params)

            try:
                run_callable(params)
            except:
                experiment.log('run_error', traceback.format_exc())

            experiment.end_run()

        grid_run(param_grid, run_callable_wrapper)

    def start_run(self):
        self.run_count = self.run_count + 1

        if self.run_id is None:
            self.run_id = self.run_count

        self.out_directory = os.path.join(self.out_directory, f'runs/{self.run_id}')

        if os.path.isdir(self.out_directory):
            shutil.rmtree(self.out_directory)

        if not os.path.isdir(self.out_directory):
            os.makedirs(self.out_directory)

    def end_run(self):
        self.out_directory = self.out_directory.replace(f'runs/{self.run_id}', '')
        self.run_id = None

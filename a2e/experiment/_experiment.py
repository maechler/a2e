import collections
import inspect
import json
import os
import pathlib
import datetime
import shutil
import traceback
import re
import numpy as np
import multiprocessing
import pandas as pd
from itertools import product
from typing import Callable, Dict, List, Union
from numpy.random import seed
from pandas import DataFrame
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, History, Callback
from a2e.automl import make_scorer
from a2e.experiment._git import git_hash, git_diff
from a2e.plotter import plot, plot_model_layer_weights
from a2e.processing.health import compute_health_score
from a2e.processing.stats import compute_reconstruction_error, mad
from a2e.utility import grid_run, synchronized


class Experiment:

    def __init__(self, experiment_id: str = None, verbose: bool = True, out_directory: str = None, auto_datetime_directory: bool = True, set_seeds: bool = True):
        self.experiment_id = experiment_id
        self.experiment_start = datetime.datetime.now()
        self.out_directory = out_directory
        self.verbose = verbose
        self.run_count = 0
        self.run_start = None
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

        self.print(f'Created new experiment "{self.experiment_id}"')
        self.log('timing.log', f'experiment_start={self.experiment_start}')
        self.log('git/hash', git_hash())
        self.log('git/diff', git_diff())

    def __exit__(self):
        self.log('timing.log', f'experiment_end={datetime.datetime.now()}')
        self.log('timing.log', f'experiment_duration={datetime.datetime.now() - self.experiment_start}')

    def _set_seeds(self):
        seed(1)
        set_seed(1)

    def log(self, key: str, value: any, mode: str = 'a'):
        self.print(f'Logging "{key}"')

        out_file_path = self._out_path(key)

        with open(out_file_path, mode) as out_file:
            if isinstance(value, dict) or isinstance(value, list):
                out_file.write(json.dumps(value, indent=2) + '\n')
            elif isinstance(value, pd.DataFrame):
                value.to_csv(out_file_path, index=True, header=True)
            else:
                out_file.write(str(value) + '\n')

    def log_plot(self, key: str, x=None, y=None, xlabel=None, ylabel=None, ylim=None, label=None, time_formatting: bool = False, create_figure: bool = True, close: bool = True):
        self.print(f'Logging plot "{key}"')

        self.log(key, y)
        plot(x=x, y=y, xlabel=xlabel, ylabel=ylabel, ylim=ylim, label=label, time_formatting=time_formatting, create_figure=create_figure, close=close, out_path=self._out_path(key))

    def log_history(self, history: History):
        self.print('Logging history')

        if 'loss' in history.history and 'val_loss' in history.history:
            self.log('metrics/val_loss', history.history['val_loss'])
            self.log_plot('metrics/loss', y=history.history['val_loss'], label='validation loss', xlabel='epoch', close=False)
            self.log_plot('metrics/loss', y=history.history['loss'], label='loss', xlabel='epoch', create_figure=False)
        else:
            if 'loss' in history.history:
                self.log_plot('metrics/loss', y=history.history['loss'], xlabel='epoch', ylabel='loss')

            if 'val_loss' in history.history:
                self.log_plot('metrics/val_loss', y=history.history['val_loss'], xlabel='epoch', ylabel='validation loss')

    def log_model(self, model: Model, key: str = 'model'):
        self.print('Logging model')

        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))

        plot_model(model, show_shapes=True, expand_nested=True, to_file=self._out_path(f'{key}/model.png'))
        self.log(f'{key}/summary', '\n'.join(model_summary))

        if len(model.get_weights()) > 0:
            plot_model_layer_weights(model, out_path=self._out_path(f'{key}/layers', is_directory=True))

    def log_predictions(self, model: Model, data_frames: Dict[str, DataFrame], pre_processing: Callable = None, pre_processing_x: Callable = None, pre_processing_y: Callable = None, rolling_window_size: int = 200, log_samples: List[int] = [0, -1], has_multiple_features: bool = False):
        self.print('Logging predictions')

        pre_processing_x = pre_processing_x if pre_processing_x is not None else pre_processing
        pre_processing_y = pre_processing_y if pre_processing_y is not None else pre_processing
        train_reconstruction_error = None

        if 'train' in data_frames:
            train_data_frame = data_frames['train']
            train_samples_x = pre_processing_x(train_data_frame) if pre_processing_x is not None else train_data_frame.to_numpy()
            train_samples_y = pre_processing_y(train_data_frame) if pre_processing_y is not None else train_data_frame.to_numpy()
            train_reconstruction_error = compute_reconstruction_error(train_samples_y, model.predict(train_samples_x), has_multiple_features=has_multiple_features)

        for key, data_frame in data_frames.items():
            samples_x = pre_processing_x(data_frame) if pre_processing_x is not None else data_frame.to_numpy()
            samples_y = pre_processing_y(data_frame) if pre_processing_y is not None else data_frame.to_numpy()

            reconstruction = model.predict(samples_x)
            reconstruction_error = compute_reconstruction_error(samples_y, reconstruction, has_multiple_features=has_multiple_features)

            if len(data_frame.index) != len(reconstruction_error):
                cut_rows = len(reconstruction_error) - len(data_frame.index)
                data_frame = data_frame.iloc[:cut_rows].copy()

            data_frame['reconstruction_error'] = reconstruction_error
            data_frame['reconstruction_error_rolling'] = data_frame['reconstruction_error'].rolling(window=rolling_window_size).median()

            self.log(f'metrics/{key}/median', np.median(reconstruction_error))
            self.log(f'metrics/{key}/mad', mad(reconstruction_error))

            self.log_plot(f'metrics/{key}/reconstruction_error', x=data_frame.index, y=reconstruction_error, label='reconstruction error', time_formatting=True, close=False)
            self.log_plot(f'metrics/{key}/reconstruction_error_rolling', x=data_frame.index, y=data_frame['reconstruction_error_rolling'], label='rolling reconstruction error', time_formatting=True, create_figure=False)

            if train_reconstruction_error is not None:
                data_frame['health_score'] = compute_health_score(train_reconstruction_error, reconstruction_error)

                self.log_plot(f'metrics/{key}/health_score', x=data_frame.index, y=data_frame['health_score'], label='health score', ylim=[0, 1], time_formatting=True, close=False)
                self.log_plot(f'metrics/{key}/health_score_rolling', x=data_frame.index, y=data_frame['health_score'].rolling(window=rolling_window_size).median(), label='rolling health score', ylim=[0, 1], time_formatting=True, create_figure=False)

            for sample_index in log_samples:
                ylim = [0, 1] if all(0.0 <= value <= 1.0 for value in samples_y[sample_index]+reconstruction[sample_index]) else None

                self.log_plot(f'metrics/{key}/samples/sample_{sample_index}', y=samples_y[sample_index], ylim=ylim, label='input', close=False)
                self.log_plot(f'metrics/{key}/samples/sample_{sample_index}', y=reconstruction[sample_index], ylim=ylim, label='reconstruction', create_figure=False)

    def _out_path(self, relative_path: str, is_directory: bool = False) -> str:
        out_file_path = pathlib.Path(os.path.join(self.out_directory, relative_path))
        out_directory = out_file_path.absolute() if is_directory else out_file_path.parent.absolute()

        if not os.path.isdir(out_directory):
            os.makedirs(out_directory)

        return str(out_file_path)

    def print(self, message: str):
        if self.verbose:
            message_prefix = f'[{self.experiment_id}]'
            message_prefix = message_prefix + f'[pid={os.getpid()}]'

            if self.run_id is not None:
                message_prefix = message_prefix + f'[run_id={self.run_id}]'

            print(f'{message_prefix} {message}')

    def callbacks(self, save_best: bool = True):
        callbacks = [
            ModelCheckpoint(self._out_path('model/model.hdf5')),
            CSVLogger(self._out_path('metrics/loss_log.csv'), separator=',', append=False),
            ExperimentCallback(self),
        ]

        if save_best:
            callbacks.append(ModelCheckpoint(
                self._out_path('model/model.best.hdf5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min')
            )

        return callbacks

    def multi_run(self, configurations: Union[dict, list], run_callable: Callable, run_id_callable: Callable = None, auto_run_id: bool = True):
        experiment = self
        number_of_runs = 0
        current_run = 0

        def run_callable_wrapper(params):
            if auto_run_id:
                self.run_id = re.sub('[(){}<>\'"]', '', '_'.join(map(str, params.values())))
            elif run_id_callable is not None:
                self.run_id = run_id_callable(params)

            nonlocal current_run
            current_run = current_run + 1

            experiment.print(f'Starting run "{self.run_id}"')
            experiment.print(f'Run {current_run} of {number_of_runs}')
            experiment.start_run()
            experiment.log('run_config', params)

            try:
                run_callable(params)
            except:
                experiment.log('run_error', traceback.format_exc())

            experiment.end_run()

        if isinstance(configurations, dict):
            number_of_runs = len(list(product(*configurations.values())))
            experiment.print(f'Number of runs: {number_of_runs}')

            grid_run(configurations, run_callable_wrapper)
        else:
            number_of_runs = len(configurations)
            experiment.print(f'Number of runs: {number_of_runs}')

            for configuration in configurations:
                run_callable_wrapper(configuration)

    def start_run(self):
        self.run_count = self.run_count + 1
        self.run_start = datetime.datetime.now()

        if self.run_id is None:
            self.run_id = self.run_count

        self.log('timing.log', f'start_run[{self.run_id}]={self.run_start}')
        self.out_directory = os.path.join(self.out_directory, f'runs/{self.run_id}')

        if os.path.isdir(self.out_directory):
            shutil.rmtree(self.out_directory)

        if not os.path.isdir(self.out_directory):
            os.makedirs(self.out_directory)

    def end_run(self):
        self.out_directory = self.out_directory.replace(f'runs/{self.run_id}', '')
        self.log('timing.log', f'end_run[{self.run_id}]={datetime.datetime.now()}')
        self.log('timing.log', f'run_duration[{self.run_id}]={datetime.datetime.now() - self.run_start}')
        self.run_id = None

    def make_scorer(self, score_func, *, greater_is_better=True, use_synchronization=True, **kwargs):
        experiment = self

        def scoring_callback(predict_scorer, score):
            nonlocal experiment
            experiment.log('scorer_history.csv', pd.DataFrame(list(predict_scorer.state['history'])), mode='w')

        return make_scorer(
            score_func,
            greater_is_better=greater_is_better,
            use_synchronization=use_synchronization,
            scoring_callbacks=[scoring_callback],
            **kwargs,
        )


class ExperimentCallback(Callback):
    def __init__(self, experiment: Experiment):
        super().__init__()

        self.experiment = experiment

    def on_epoch_end(self, epoch, logs={}):
        epochs = self.params['epochs']
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')

        if loss is not None:
            loss = '{:.4f}'.format(loss)

        if val_loss is not None:
            val_loss = '{:.4f}'.format(val_loss)

        self.experiment.print(f'Epoch {epoch+1} of {epochs} end: loss={loss}, val_loss={val_loss}')

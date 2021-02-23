import inspect
import json
import os
import pathlib
import datetime
import shutil
import traceback
import re
import numpy as np
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from statistics import median
from itertools import product
from typing import Callable, Dict, List, Union
from numpy import percentile
from numpy.random import seed
from pandas import DataFrame, Series
from tensorflow.python.framework.random_seed import set_seed
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, History, Callback
from tensorflow.python.keras.models import load_model
from a2e.experiment._git import git_hash, git_diff
from a2e.model import KerasModel
from a2e.optimizer import OptimizationResult
from a2e.plotter import plot, plot_model_layer_weights, plot_model_layer_activations, plot_roc
from a2e.processing.health import compute_health_score
from a2e.processing.stats import compute_reconstruction_error, mad
from a2e.utility import grid_run, compute_roc, compute_classification_metrics, z_score

warnings.simplefilter(action='ignore', category=Warning)


class Experiment:

    def __init__(self, experiment_id: str = None, verbose: bool = True, out_directory: str = None, auto_datetime_directory: bool = True, set_seeds: bool = True, suppress_warnings: bool = True):
        self.experiment_id = experiment_id
        self.experiment_start = datetime.datetime.now()
        self.out_directory = out_directory
        self.verbose = verbose
        self.run_count = 0
        self.run_start = None
        self.run_id = None

        if set_seeds:
            seed(1)
            set_seed(1)

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

        try:
            self.log('git/hash', git_hash())
            self.log('git/state.diff', git_diff())
        except:
            pass  # not a git repo

        if suppress_warnings:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')
            logging.disable()
        else:
            warnings.simplefilter(action='default', category=Warning)

    def __exit__(self):
        self.log('timing.log', f'experiment_end={datetime.datetime.now()}')
        self.log('timing.log', f'experiment_duration={datetime.datetime.now() - self.experiment_start}')

    def _add_optional_extension(self, file_name: str, extension: str):
        if '.' in file_name:
            return file_name
        else:
            return f'{file_name}.{extension}'

    def log(self, key: str, value: any, mode: str = 'a', to_pickle=False):
        self.print(f'Logging "{key}"')

        if to_pickle:
            key = self._add_optional_extension(key, 'pickle')
            mode = mode + 'b'
        elif isinstance(value, dict) or isinstance(value, list):
            key = self._add_optional_extension(key, 'json')
        elif isinstance(value, pd.DataFrame):
            key = self._add_optional_extension(key, 'csv')
        else:
            key = self._add_optional_extension(key, 'txt')

        out_file_path = self._out_path(key)

        with open(out_file_path, mode) as out_file:
            if to_pickle:
                pickle.dump(value, out_file)
            elif isinstance(value, dict) or isinstance(value, list):
                out_file.write(json.dumps(value, indent=2) + '\n')
            elif isinstance(value, pd.DataFrame):
                value.to_csv(out_file_path, index=True, header=True)
            else:
                out_file.write(str(value) + '\n')

    def plot_roc(self, key, fpr, tpr, close=True, create_figure=True):
        self.print(f'Logging roc plot "{key}"')

        plot_roc(fpr, tpr, out_path=self._out_path(key), close=close, create_figure=create_figure)

    def plot(self, key: str, *args, **kwargs):
        self.print(f'Logging plot "{key}"')

        plot(*args, **kwargs, out_path=self._out_path(key))

    def log_history(self, history: History, key: str = None):
        self.print('Logging history')
        log_base_path = 'training/history' if key is None else f'training/history/{key}'

        if 'loss' in history.history and 'val_loss' in history.history:
            self.plot(f'{log_base_path}/loss', y=history.history['val_loss'], label='validation loss', xlabel='epoch', close=False)
            self.plot(f'{log_base_path}/loss', y=history.history['loss'], label='loss', xlabel='epoch', create_figure=False)
        else:
            if 'loss' in history.history:
                self.plot(f'{log_base_path}/loss', y=history.history['loss'], xlabel='epoch', ylabel='loss')

            if 'val_loss' in history.history:
                self.plot(f'{log_base_path}/val_loss', y=history.history['val_loss'], xlabel='epoch', ylabel='validation loss')

        self.log(f'{log_base_path}/history', pd.DataFrame.from_dict(history.history))

    def log_keras_model(self, model: Model, key: str = None):
        self.print('Logging model')

        log_base_path = 'model' if key is None else f'model/{key}'
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))

        plot_model(model, show_shapes=True, expand_nested=True, to_file=self._out_path(f'{log_base_path}/model.png'))
        self.log(f'{log_base_path}/summary', '\n'.join(model_summary))

        if len(model.get_weights()) > 0:
            plot_model_layer_weights(model, out_path=self._out_path(f'{log_base_path}/weights', is_directory=True))
            model.save(self._out_path(f'{log_base_path}/model.h5'))

    def log_optimization_result(self, optimization_result: OptimizationResult):
        evaluation_results = optimization_result.evaluation_results
        evaluation_results_without_outliers = evaluation_results[((evaluation_results['cost'] - evaluation_results['cost'].median()) / mad(evaluation_results['cost'].dropna())).abs() < 3]

        self.log('search/history', evaluation_results, mode='w')
        self.log('search/best_configuration', optimization_result.best_config())
        self.log('search/average_configuration', optimization_result.config_by_percentile_rank(0.5))
        self.log('search/worst_10th_configuration', optimization_result.config_by_percentile_rank(0.1))
        self.log('search/worst_configuration', optimization_result.config_by_percentile_rank(0.0))

        self.plot('search/best_model_id', y=evaluation_results['best_evaluation_id'], xlabel='iteration', ylabel='best evaluation ID')
        self.plot('search/best_cost', y=evaluation_results['best_cost'], xlabel='iteration', ylabel='best cost')

        self.plot('search/cost', y=evaluation_results['cost'], xlabel='iteration', ylabel='cost')
        evaluation_results.plot(subplots=True, figsize=(10, 100))
        plt.gcf().savefig(self._out_path('search/history.pdf'), format='pdf')

        self.plot('search/cost_cleaned', y=evaluation_results_without_outliers['cost'], xlabel='iteration', ylabel='cost')
        evaluation_results_without_outliers.plot(subplots=True, figsize=(10, len(evaluation_results.columns) * 5))
        plt.gcf().savefig(self._out_path('search/history_cleaned.pdf'), format='pdf')

    def log_keras_predictions(self, model: Union[Model, KerasModel], data_frames: Dict[str, DataFrame], labels: Dict[str, Series] = None, key: str = None, pre_processing: Callable = None, pre_processing_x: Callable = None, pre_processing_y: Callable = None, rolling_window_size: int = 200, log_samples: List[int] = [0, -1], has_multiple_features: bool = False, threshold_percentile: int = 99):
        self.print('Logging predictions')

        log_base_path = 'predictions' if key is None else f'predictions/{key}'
        pre_processing_x = pre_processing_x if pre_processing_x is not None else pre_processing
        pre_processing_y = pre_processing_y if pre_processing_y is not None else pre_processing
        train_reconstruction_error = None
        train_reconstruction_error_rolling = None
        train_z_scores = None
        train_z_scores_rolling = None

        if 'train' in data_frames:
            train_data_frame = data_frames['train']
            train_samples_x = pre_processing_x(train_data_frame) if pre_processing_x is not None else train_data_frame.to_numpy()
            train_samples_y = pre_processing_y(train_data_frame) if pre_processing_y is not None else train_data_frame.to_numpy()

            train_reconstruction_error = compute_reconstruction_error(train_samples_y, model.predict(train_samples_x), has_multiple_features=has_multiple_features)
            train_reconstruction_error_median = median(train_reconstruction_error)
            train_reconstruction_error_mad = mad(train_reconstruction_error)
            train_z_scores = list(map(lambda x: z_score(x, train_reconstruction_error_median, train_reconstruction_error_mad), train_reconstruction_error))

            train_reconstruction_error_rolling = np.convolve(train_reconstruction_error, np.ones(rolling_window_size)/rolling_window_size, mode='valid')
            train_reconstruction_error_rolling_median = median(train_reconstruction_error_rolling)
            train_reconstruction_error_rolling_mad = mad(train_reconstruction_error_rolling)
            train_z_scores_rolling = list(map(lambda x: z_score(x, train_reconstruction_error_rolling_median, train_reconstruction_error_rolling_mad), train_reconstruction_error_rolling))

        for data_frame_key, data_frame in data_frames.items():
            data_frame = data_frame.copy()
            data_frame_log_path = f'{log_base_path}/{data_frame_key}'

            try:
                samples_x = pre_processing_x(data_frame) if pre_processing_x is not None else data_frame.to_numpy()
                samples_y = pre_processing_y(data_frame) if pre_processing_y is not None else data_frame.to_numpy()

                reconstruction = model.predict(samples_x)
                reconstruction_error = compute_reconstruction_error(samples_y, reconstruction, has_multiple_features=has_multiple_features)

                if len(data_frame.index) != len(reconstruction_error):
                    cut_rows = len(reconstruction_error) - len(data_frame.index)
                    data_frame = data_frame.iloc[:cut_rows].copy()

                data_frame['reconstruction_error'] = reconstruction_error
                data_frame['reconstruction_error_rolling'] = data_frame['reconstruction_error'].rolling(window=rolling_window_size).median().fillna(method='backfill')
                data_frame['z_score'] = list(map(lambda x: z_score(x, train_reconstruction_error_median, train_reconstruction_error_mad), reconstruction_error))
                data_frame['z_score_rolling'] = list(map(lambda x: z_score(x, train_reconstruction_error_rolling_median, train_reconstruction_error_rolling_mad), data_frame['reconstruction_error_rolling'].values))

                self.plot(f'{data_frame_log_path}/reconstruction_error', x=data_frame.index, y=reconstruction_error, label='reconstruction error', time_formatting=True, close=False)
                self.plot(f'{data_frame_log_path}/reconstruction_error_rolling', x=data_frame.index, y=data_frame['reconstruction_error_rolling'], label='rolling reconstruction error', time_formatting=True, create_figure=False)

                self.plot(f'{data_frame_log_path}/z_score', x=data_frame.index, y=data_frame['z_score'], label='z-score', time_formatting=True, close=False)
                self.plot(f'{data_frame_log_path}/z_score_rolling', x=data_frame.index, y=data_frame['z_score_rolling'], label='rolling z-score', time_formatting=True, create_figure=False)

                if labels is not None and data_frame_key in labels and len(set(labels[data_frame_key].values)) > 1:
                    roc = compute_roc(labels[data_frame_key].values, reconstruction_error)
                    roc_rolling = compute_roc(labels[data_frame_key].values, data_frame['reconstruction_error_rolling'].values)

                    self.log(f'{data_frame_log_path}/roc/auc', roc['auc'])
                    self.log(f'{data_frame_log_path}/roc/data', roc, to_pickle=True)
                    self.plot_roc(f'{data_frame_log_path}/roc/fpr_tpr', roc['fpr'], roc['tpr'])

                    self.log(f'{data_frame_log_path}/roc/auc_rolling', roc_rolling['auc'])
                    self.log(f'{data_frame_log_path}/roc/data_rolling', roc_rolling, to_pickle=True)
                    self.plot_roc(f'{data_frame_log_path}/roc/fpr_tpr_rolling', roc_rolling['fpr'], roc_rolling['tpr'])

                if train_reconstruction_error is not None:
                    data_frame['health_score'] = compute_health_score(train_reconstruction_error, reconstruction_error)
                    data_frame['health_score_rolling'] = data_frame['health_score'].rolling(window=rolling_window_size).median().fillna(method='backfill')
                    log_metrics = ['reconstruction_error', 'reconstruction_error_rolling', 'health_score', 'health_score_rolling']

                    if labels is not None and data_frame_key in labels and len(set(labels[data_frame_key].values)) > 1:
                        threshold = percentile(train_z_scores, threshold_percentile)
                        rolling_threshold = percentile(train_z_scores_rolling, threshold_percentile)
                        prediction = (data_frame['z_score'] > threshold).astype(int)
                        prediction_rolling = (data_frame['z_score_rolling'] > rolling_threshold).astype(int)
                        metrics = compute_classification_metrics(labels[data_frame_key].values, prediction)
                        metrics_rolling = compute_classification_metrics(labels[data_frame_key].values, prediction_rolling)

                        self.log(f'{data_frame_log_path}/classification/thresholds', {'threshold_percentile': threshold_percentile, 'threshold': threshold, 'rolling_threshold': rolling_threshold})
                        self.log(f'{data_frame_log_path}/classification/metrics', metrics)
                        self.log(f'{data_frame_log_path}/classification/metrics_rolling', metrics_rolling)
                        log_metrics.append('z_score')
                        log_metrics.append('z_score_rolling')

                    self.plot(f'{data_frame_log_path}/health_score/health_score', x=data_frame.index, y=data_frame['health_score'], label='health score', ylim=[0, 1], time_formatting=True, close=False)
                    self.plot(f'{data_frame_log_path}/health_score/health_score_rolling', x=data_frame.index, y=data_frame['health_score'].rolling(window=rolling_window_size).median().fillna(method='backfill'), label='rolling health score', ylim=[0, 1], time_formatting=True, create_figure=False)

                    self.log(f'{data_frame_log_path}/metrics', data_frame[log_metrics])
                else:
                    self.log(f'{data_frame_log_path}/metrics', data_frame[['reconstruction_error', 'reconstruction_error_rolling', 'z_score', 'z_score_rolling']])

                for sample_index in log_samples:
                    ylim = [0, 1] if all(0.0 <= value <= 1.0 for value in samples_y[sample_index]+reconstruction[sample_index]) else None
                    sample_log_path = f'{data_frame_log_path}/samples/sample_{sample_index}'

                    self.plot(f'{sample_log_path}/input', y=samples_y[sample_index], ylim=ylim, label='input', close=False)
                    self.plot(f'{sample_log_path}/reconstruction', y=reconstruction[sample_index], ylim=ylim, label='reconstruction', create_figure=False)

                    self.log(f'{sample_log_path}/data', pd.DataFrame.from_dict({'input': samples_y[sample_index], 'reconstruction': reconstruction[sample_index]}))

                    plot_model_layer_activations(model=model if isinstance(model, Model) else model.model, sample=samples_x[sample_index], out_path=self._out_path(f'{sample_log_path}/activations/', is_directory=True))
            except ValueError:
                self.log(f'{data_frame_log_path}/error', traceback.format_exc())

    def _out_path(self, relative_path: str, is_directory: bool = False) -> str:
        out_file_path = pathlib.Path(os.path.join(self.out_directory, relative_path))
        out_directory = out_file_path.absolute() if is_directory else out_file_path.parent.absolute()

        if not os.path.isdir(out_directory):
            os.makedirs(out_directory)

        return str(out_file_path)

    def print(self, message: str):
        if self.verbose:
            datetime_string = datetime.datetime.now().strftime('%d.%m.%Y %H:%M:%S')
            message_prefix = f'[{datetime_string}][{self.experiment_id}]'
            #message_prefix = message_prefix + f'[pid={os.getpid()}]'

            if self.run_id is not None:
                message_prefix = message_prefix + f'[run_id={self.run_id}]'

            print(f'{message_prefix} {message}')

    def keras_callbacks(self, save_best: bool = True):
        callbacks = [
            ModelCheckpoint(self._out_path('training/callbacks/model.current.h5')),
            CSVLogger(self._out_path('training/callbacks/loss_log.csv'), separator=',', append=False),
            ExperimentCallback(self),
        ]

        if save_best:
            callbacks.append(ModelCheckpoint(
                self._out_path('training/callbacks/model.best.h5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min')
            )

        return callbacks

    def load_best_model(self) -> Model:
        return load_model(self._out_path('training/callbacks/model.best.h5'))

    def multi_run(self, configurations: Union[dict, list], run_callable: Callable, run_id_callable: Callable = None, auto_run_id: bool = True):
        experiment = self
        number_of_runs = 0
        current_run = 0

        def run_callable_wrapper(params):
            if auto_run_id:
                self.run_id = re.sub(r'[(){}<>\',\s"]', '', '/'.join(map(str, params.values())))
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

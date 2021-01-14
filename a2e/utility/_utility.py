import sys
import importlib
import numpy as np
from math import isnan, isinf
from statistics import median
from datetime import datetime, timezone, tzinfo
from itertools import product
from typing import Callable, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix, roc_curve, roc_auc_score
from a2e.processing.stats import mad


def timestamp_to_date_time(timestamp: float, tz: tzinfo = timezone.utc) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=tz)


def grid_run(param_grid: dict, run_callable: Callable):
    param_grid_keys = list(param_grid)
    param_grid_values = param_grid.values()

    for param_comb in product(*param_grid_values):
        param_dict = {}

        for i, param_value in enumerate(param_comb):
            param_name = param_grid_keys[i]
            param_dict[param_name] = param_value

        run_callable(param_dict)


def build_samples(data: np.ndarray, target_sample_length: int, target_dimensions: int = 2):
    if len(data.shape) == 1:
        drop_data = len(data) % target_sample_length
        samples = data[:-drop_data] if drop_data > 0 else data
    else:
        input_sample_length = data.shape[1]
        drop_data = input_sample_length % target_sample_length
        drop_start = input_sample_length - drop_data
        samples = np.delete(data, np.s_[drop_start:input_sample_length], axis=1)

    return samples.reshape(-1, target_sample_length) if target_dimensions == 2 else samples.reshape(-1, target_sample_length, 1)


def load_from_module(identifier: str) -> any:
    module_name, item_name = identifier.rsplit('.', 1)

    return getattr(importlib.import_module(module_name), item_name)


def synchronized(method: Callable):
    def synchronized_method(self, *args, **kwargs):
        if hasattr(self, '_lock'):
            with self._lock:
                return method(self, *args, **kwargs)
        else:
            return method(self, *args, **kwargs)

    return synchronized_method


def z_score(value: Union[list, np.array, float], given_median: Union[None, float] = None, given_mad: Union[None, float] = None):
    if isinstance(value, (list, np.ndarray)):
        if given_median is None:
            given_median = median(value)
        if given_mad is None:
            given_mad = mad(value)

        return list(map(lambda x: z_score(x, given_median, given_mad), value))
    else:
        return abs(value - given_median) / given_mad


def compute_roc(y_true, y_pred, target_format='{:.4f}'):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return {
        'auc': float(target_format.format(auc)) if target_format is not None else auc,
        'thresholds': thresholds.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    }


def compute_classification_metrics(y_true, y_pred, average='binary', target_format='{:.4f}'):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred, average=average)
    matthews_cc = matthews_corrcoef(y_true, y_pred)
    confusion_matrix_results = confusion_matrix(y_true, y_pred).tolist()

    return {
        'accuracy': float(target_format.format(accuracy)) if target_format is not None else accuracy,
        'precision': float(target_format.format(precision)) if target_format is not None else precision,
        'recall': float(target_format.format(recall)) if target_format is not None else recall,
        'f_score': float(target_format.format(f_score)) if target_format is not None else f_score,
        'matthews_cc': float(target_format.format(matthews_cc)) if target_format is not None else matthews_cc,
        'confusion_matrix': {
            'true_negatives': confusion_matrix_results[0][0],
            'false_negatives': confusion_matrix_results[1][0],
            'true_positives': confusion_matrix_results[1][1],
            'false_positives': confusion_matrix_results[0][1],
        },
    }


def inf_nan_to_float_max(number: float) -> float:
    if isnan(number) or isinf(number):
        return sys.float_info.max

    return number

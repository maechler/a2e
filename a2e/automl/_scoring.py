import numpy as np
from statistics import median
from sklearn.metrics._scorer import _BaseScorer

from a2e.models._feed_forward import compute_model_compression
from a2e.processing.health import compute_health_score
from a2e.processing.stats import compute_reconstruction_error


class ScoreInfo:
    def __init__(self, score: float, score_metrics: dict = {}):
        self.score = score
        self.score_metrics = score_metrics


class KerasPredictScorer(_BaseScorer):
    def __init__(self, score_func, greater_is_better=True, **kwargs):
        self.estimator_search = None
        sign = 1 if greater_is_better else -1

        super().__init__(score_func, sign, kwargs)

    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        # store values because they get lost when the prediction is carried out
        estimator_history = estimator.model.history
        estimator_sk_params = estimator.sk_params
        y_pred = method_caller(estimator, 'predict', X)

        score_info = self._score_func(y_true, y_pred, estimator=estimator, estimator_history=estimator_history, **self._kwargs)

        if self.estimator_search is not None:
            self.estimator_search.add_score_to_history(score_info, estimator_sk_params)

        return self._sign * score_info.score


def f1_loss_compression_score(y_true, y_pred, estimator, estimator_history,  **kwargs):
    history = estimator_history.history
    model = estimator.model

    val_loss_start = history['val_loss'][0]
    val_loss = min(history['val_loss'][-1], 1)

    compression = compute_model_compression(model)
    loss_improvement = 1 - (val_loss / val_loss_start)

    f1_score = 2 * (compression * loss_improvement) / (compression + loss_improvement)
    score_metrics = {
        'compression': compression,
        'loss_improvement': loss_improvement,
    }

    return ScoreInfo(f1_score, score_metrics)


def val_loss_score(y_true, y_pred, estimator, estimator_history, **kwargs):
    history = estimator_history.history

    if 'val_loss' not in history:
        raise ValueError('No val_loss provided, try to add validation_split=x.x to KerasEstimator constructor')

    return ScoreInfo(min(history['val_loss']))


def reconstruction_error_score(y_true, y_pred, estimator, estimator_history, **kwargs):
    return ScoreInfo(median(compute_reconstruction_error(y_true, y_pred)))


def reconstruction_error_compression_score(y_true, y_pred, estimator, estimator_history, **kwargs):
    reconstruction_error = median(compute_reconstruction_error(y_true, y_pred))
    compression = compute_model_compression(estimator.model)
    score = reconstruction_error / compression

    return ScoreInfo(score, {
        'reconstruction_error': reconstruction_error,
        'compression': compression,
    })


def loss_score(y_true, y_pred, estimator, estimator_history, **kwargs):
    history = estimator_history.history

    return ScoreInfo(min(history['loss']))


def health_score(y_true, y_pred, estimator, estimator_history, **kwargs):
    reconstruction_error = compute_reconstruction_error(y_true, y_pred)
    health_score = compute_health_score(reconstruction_error, reconstruction_error, mode='median')
    score = float(np.median(health_score))

    return ScoreInfo(score)


def min_health_score(y_true, y_pred, estimator, estimator_history, rolling_window_size=200, **kwargs):
    reconstruction_error = compute_reconstruction_error(y_true, y_pred)
    health_score = compute_health_score(reconstruction_error, reconstruction_error, mode='median')
    rolling_health_score = np.convolve(health_score, np.ones(rolling_window_size)/rolling_window_size, mode='valid')
    score = float(np.min(rolling_health_score))

    return ScoreInfo(score)

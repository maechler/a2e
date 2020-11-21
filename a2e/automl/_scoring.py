import collections
import multiprocessing
from typing import Callable, List
from sklearn.metrics._scorer import _PredictScorer
from a2e.utility import synchronized


class PredictScorer(_PredictScorer):

    def __init__(self, score_func, sign, kwargs, use_synchronization: bool = True, scoring_callbacks: List[Callable] = []):
        super().__init__(score_func, sign, kwargs)

        if use_synchronization:
            process_manager = multiprocessing.Manager()
            self._lock = process_manager.RLock()
            self.state = process_manager.dict()
            self.state['history'] = process_manager.list()
        else:
            self.state = {
                'history': []
            }

        self.state['model_id'] = 0
        self.scoring_callbacks = scoring_callbacks

    @synchronized
    def add_scoring_to_history(self, scoring: dict):
        self.state['model_id'] = self.state['model_id'] + 1
        score = scoring['score']

        if len(self.state['history']) == 0:
            best_model_row_id = 0
            best_model_row_score = -1
        else:
            best_model_row_score = 0
            best_model_row_id = 0

            for history in self.state['history']:
                if history['score'] > best_model_row_score:
                    best_model_row_score = history['score']
                    best_model_row_id = history['model_id']

        history_row = {
            'model_id': self.state['model_id'],
            'best_model_id': self.state['model_id'] if score > best_model_row_score else best_model_row_id,
            'score': score,
        }

        if 'score_metrics' in scoring:
            for key, value in scoring['score_metrics'].items():
                history_row[key] = value

        for key, value in self._kwargs['a2e']['estimator'].sk_params.items():
            history_row[key] = value

        self.state['history'].append(history_row)

    @synchronized
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        # Store estimator and history in order to use it later in the scoring functions
        self._kwargs['a2e']['estimator'] = estimator
        self._kwargs['a2e']['estimator_history'] = estimator.model.history

        score = super()._score(method_caller, estimator, X, y_true, sample_weight)

        self.add_scoring_to_history(self._kwargs['a2e']['scoring'])

        for scoring_callback in self.scoring_callbacks:
            scoring_callback(self, score)

        return score


def make_scorer(score_func, *, greater_is_better=True, use_synchronization: bool = True, scoring_callbacks: List[Callable] = [], **kwargs):
    sign = 1 if greater_is_better else -1

    if 'a2e' not in kwargs:
        kwargs['a2e'] = collections.defaultdict(dict)

    return PredictScorer(
        score_func,
        sign,
        kwargs,
        use_synchronization=use_synchronization,
        scoring_callbacks=scoring_callbacks,
    )


def f1_loss_compression_score(y_true, y_pred, **kwargs):
    estimator = kwargs['a2e']['estimator']
    history = kwargs['a2e']['estimator_history'].history
    sk_params = estimator.sk_params
    model = estimator.model

    val_loss_start = history['loss'][0] #val_loss?
    val_loss = min(history['loss'][-1], 1)
    input_dimension = model.input_shape[1]
    encoding_dimension = sk_params['encoding_dimension']

    compression = 1 - (encoding_dimension / input_dimension)
    loss_improvement = 1 - (val_loss / val_loss_start)

    f1_score = 2 * (compression * loss_improvement) / (compression + loss_improvement)

    kwargs['a2e']['scoring'] = {
        'score': f1_score,
        'score_metrics': {
            'compression': compression,
            'loss_improvement': loss_improvement,
        }
    }

    return f1_score

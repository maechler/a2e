from sklearn.metrics._scorer import _PredictScorer


class PredictScorer(_PredictScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        # Store estimator and history in order to use it later in the scoring functions
        self._kwargs['a2e']['estimator'] = estimator
        self._kwargs['a2e']['history'] = estimator.model.history

        return super()._score(method_caller, estimator, X, y_true, sample_weight)


def make_scorer(score_func, *, greater_is_better=True, **kwargs):
    sign = 1 if greater_is_better else -1

    return PredictScorer(score_func, sign, kwargs)


def f1_loss_compression_score(y_true, y_pred, **kwargs):
    estimator = kwargs['a2e']['estimator']
    history = kwargs['a2e']['history'].history
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

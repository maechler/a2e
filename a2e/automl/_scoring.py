from sklearn.metrics._scorer import _PredictScorer


class MyPredictScorer(_PredictScorer):
    def _score(self, method_caller, estimator, X, y_true, sample_weight=None):
        self._kwargs['estimator'] = estimator

        return super()._score(method_caller, estimator, X, y_true, sample_weight)


def my_make_scorer(score_func, *, greater_is_better=True, **kwargs):
    sign = 1 if greater_is_better else -1

    return MyPredictScorer(score_func, sign, kwargs)


def f1_loss_compression_score(y_true, y_pred, **kwargs):
    estimator = kwargs['estimator']
    sk_params = estimator.sk_params
    val_loss_start = estimator.model.history.history['val_loss'][0]
    val_loss = min(estimator.model.history.history['val_loss'][-1], 1)
    input_dimension = kwargs['estimator'].model.input_shape[1]
    encoding_dimension = sk_params['encoding_dimension']

    compression = 1 - (encoding_dimension / input_dimension)
    loss_improvement = 1 - (val_loss / val_loss_start)

    f1_score = 2 * (compression * loss_improvement) / (compression + loss_improvement)

    return f1_score

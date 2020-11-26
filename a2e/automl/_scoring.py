from sklearn.metrics._scorer import _BaseScorer


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
        y_pred = method_caller(estimator, "predict", X)

        score, score_info = self._score_func(y_true, y_pred, estimator=estimator, estimator_history=estimator_history, **self._kwargs)

        if self.estimator_search is not None:
            self.estimator_search.add_score_to_history(score_info, estimator_sk_params)

        return self._sign * score


def f1_loss_compression_score(y_true, y_pred, estimator, estimator_history,  **kwargs):
    history = estimator_history.history
    sk_params = estimator.sk_params
    model = estimator.model

    val_loss_start = history['loss'][0]
    val_loss = min(history['loss'][-1], 1)
    input_dimension = model.input_shape[1]
    encoding_dimension = sk_params['encoding_dimension']

    compression = 1 - (encoding_dimension / input_dimension)
    loss_improvement = 1 - (val_loss / val_loss_start)

    f1_score = 2 * (compression * loss_improvement) / (compression + loss_improvement)
    score_metrics = {
        'compression': compression,
        'loss_improvement': loss_improvement,
    }

    return f1_score, ScoreInfo(f1_score, score_metrics)

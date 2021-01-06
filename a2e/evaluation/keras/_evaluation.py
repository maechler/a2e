from statistics import median
from tensorflow.python.keras.models import Model
from a2e.evaluation import EvaluationResult
from a2e.model.keras import compute_model_compression
from a2e.processing.stats import compute_reconstruction_error


def loss_cost(config, y_true, y_pred, model: Model, history: dict, **kwargs) -> EvaluationResult:
    if 'loss' not in history or len(history['loss']) == 0:
        raise ValueError('History must contain loss in order to use loss_cost function')

    return EvaluationResult(
        cost=history['loss'][-1],
        config=config,
    )


def val_loss_cost(config, y_true, y_pred, model: Model, history: dict, **kwargs) -> EvaluationResult:
    if 'val_loss' not in history or len(history['val_loss']) == 0:
        raise ValueError('History must contain val_loss in order to use val_loss_cost function')

    return EvaluationResult(
        cost=history['val_loss'][-1],
        config=config,
    )


def val_loss_vs_compression_cost(config, y_true, y_pred, model: Model, history: dict, **kwargs) -> EvaluationResult:
    if 'val_loss' not in history or len(history['val_loss']) == 0:
        raise ValueError('History must contain val_loss in order to use val_loss_vs_compression_cost function')

    val_loss = history['val_loss'][-1]
    compression = compute_model_compression(model)
    cost = val_loss / compression

    return EvaluationResult(
        cost=cost,
        config=config,
        info={
            'compression': compression,
        }
    )


def reconstruction_error_vs_compression_cost(config, y_true, y_pred, model: Model, history: dict, **kwargs) -> EvaluationResult:
    compression = compute_model_compression(model)
    reconstruction_error = median(compute_reconstruction_error(y_true, y_pred))
    cost = reconstruction_error / compression

    return EvaluationResult(
        cost=cost,
        config=config,
        info={
            'compression': compression,
            'reconstruction_error': reconstruction_error,
        }
    )
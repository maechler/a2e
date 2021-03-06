import numpy as np
from a2e.evaluation import EvaluationResult
from a2e.processing.health import compute_health_score
from a2e.processing.stats import compute_reconstruction_error
from a2e.utility import z_score


def reconstruction_error_cost(config, y_true, y_pred, **kwargs) -> EvaluationResult:
    reconstruction_errors = compute_reconstruction_error(y_true, y_pred)
    cost = np.sqrt(np.average(np.power(reconstruction_errors, 2)))

    return EvaluationResult(
        cost=cost,
        config=config,
    )


def uniform_reconstruction_error_cost(config, y_true, y_pred, **kwargs) -> EvaluationResult:
    reconstruction_errors = compute_reconstruction_error(y_true, y_pred)
    reconstruction_error_cost = np.sqrt(np.average(np.power(reconstruction_errors, 2)))

    reconstruction_error_z_scores = z_score(reconstruction_errors)
    reconstruction_error_z_score_cost = np.sqrt(np.average(np.power(reconstruction_error_z_scores, 2)))

    cost = reconstruction_error_cost * reconstruction_error_z_score_cost

    return EvaluationResult(
        cost=cost,
        config=config,
        info={
            'reconstruction_error_cost': reconstruction_error_cost,
            'reconstruction_error_z_score_cost': reconstruction_error_z_score_cost,
        }
    )


def health_score_cost(config, y_true, y_pred, **kwargs) -> EvaluationResult:
    reconstruction_errors = compute_reconstruction_error(y_true, y_pred)
    health_scores = compute_health_score(reconstruction_errors, reconstruction_errors, mode='median')
    health_score = float(np.median(health_scores))
    cost = 1 - health_score

    return EvaluationResult(
        cost=cost,
        config=config,
        info={
            'health_score': health_score,
        }
    )


def min_health_score_cost(config, y_true, y_pred, rolling_window_size=200, **kwargs) -> EvaluationResult:
    reconstruction_errors = compute_reconstruction_error(y_true, y_pred)
    health_scores = compute_health_score(reconstruction_errors, reconstruction_errors, mode='median')
    rolling_health_scores = np.convolve(health_scores, np.ones(rolling_window_size)/rolling_window_size, mode='valid')
    min_health_score = float(np.min(rolling_health_scores))
    cost = 1 - min_health_score

    return EvaluationResult(
        cost=cost,
        config=config,
        info={
            'min_health_score': min_health_score,
        }
    )

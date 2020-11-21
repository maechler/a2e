"""
The :mod:`a2e.automl` provides scoring functions and
"""

from ._scoring import f1_loss_compression_score
from ._scoring import make_scorer
from ._scoring import PredictScorer

__all__ = [
    'f1_loss_compression_score',
    'make_scorer',
    'PredictScorer',
]

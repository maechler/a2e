"""
The :mod:`a2e.plotter` provides utility functions to plot data and models
"""

from ._plotter import plot
from ._plotter import plot_model_layer_weights
from ._plotter import plot_model_layer_activations
from ._plotter import plot_roc

__all__ = [
    'plot',
    'plot_model_layer_weights',
    'plot_model_layer_activations',
    'plot_roc',
]

"""
The :mod:`a2e.models` provides different autoencoder models
"""

from ._feed_forward import create_feed_forward_autoencoder
from ._conv import create_conv_dense_autoencoder
from ._conv import create_conv_max_pool_autoencoder
from ._conv import create_conv_transpose_autoencoder
from ._lstm import create_lstm_autoencoder
from ._lstm import create_lstm_to_dense_autoencoder

__all__ = [
    'create_feed_forward_autoencoder',
    'create_conv_dense_autoencoder',
    'create_conv_max_pool_autoencoder',
    'create_conv_transpose_autoencoder',
    'create_lstm_autoencoder',
    'create_lstm_to_dense_autoencoder',
]

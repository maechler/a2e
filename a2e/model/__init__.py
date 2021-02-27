"""
The module `a2e.model` provides an abstract base model and a concrete Keras model
"""

from a2e.model._abstract_model import AbstractModel
from a2e.model._keras_model import KerasModel

__all__ = [
    'AbstractModel',
    'KerasModel',
]

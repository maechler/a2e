"""
The :mod:`a2e.algorithm` provides an abstract algorithm step and different algorithm implementations
"""

from ._abstract_algorithm import AbstractAlgorithm
from ._void import VoidAlgorithm
from ._keras import KerasFitAlgorithm
from ._keras import KerasPredictionAlgorithm
from ._scikit import ScikitSearchAlgorithm

import math


def health_score_sigmoid(x: float, shrink: float = 0.75, shift: float = 5.0):
    return -(math.exp(shrink * x - shift) / (math.exp(shrink * x - shift) + 1)) + 1

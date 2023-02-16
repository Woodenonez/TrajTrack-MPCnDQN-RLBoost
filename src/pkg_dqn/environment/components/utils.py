"""
Component utils
"""

import numpy as np
import numpy.typing as npt
from typing import Union


def normalize_distance(distance: Union[npt.NDArray, float], max_distance: float = 10) -> Union[npt.NDArray, float]:
    """
    Normalizes a positive distance to be in the interval [0, 1], see the
    accompanying report for details
    """
    return 2 / (1 + np.exp(-2 * distance / max_distance)) - 1

def normalize(x: Union[npt.NDArray, float], min: float, max: float) -> Union[npt.NDArray, float]:
    """
    Normalize an arbitrary value bounded by ``min`` and ``max`` to be in the
    interval [0, 1]
    """
    return 2 * (x - min) / (max - min) - 1

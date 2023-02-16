import numpy as np

from numpy.typing import ArrayLike

class Goal:
    """
    Describes the position of the reference path goal
    """
    def __init__(self, position: ArrayLike):
        self.position = np.asarray(position, dtype=np.float32)

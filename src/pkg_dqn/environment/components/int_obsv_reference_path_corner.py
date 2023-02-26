import numpy.typing as npt
import numpy as np
from math import atan2, cos, sin
from . import Component
from .utils import normalize_distance


class ReferencePathCornerObservation(Component):
    """
    Observes the distance and relative angle to some upcoming corners along the
    reference path
    """
    def __init__(self, num_samples: int):
        """
        :param num_samples: The number of observed points
        """
        self.num_samples = num_samples

        self.internal_obs_min = [-1, -1, 0] * self.num_samples
        self.internal_obs_max = [1, 1, 1] * self.num_samples

    def internal_obs(self) -> npt.ArrayLike:
        obs = np.zeros((3 * self.num_samples,), dtype=np.float32)
        length = 0
        i = 0

        while length < self.env.path_progress:
            length += np.linalg.norm(np.asarray(self.env.path.coords[i + 1]) - self.env.path.coords[i])
            i += 1

        for j in range(self.num_samples):
            i = min(len(self.env.path.coords) - 1, i)

            point = np.asarray(self.env.path.coords[i])
            delta = point - self.env.agent.position

            absulute_point_angle = atan2(delta[1], delta[0])
            relative_point_angle = absulute_point_angle - self.env.agent.angle

            obs[3 * j + 0] = cos(relative_point_angle)
            obs[3 * j + 1] = sin(relative_point_angle)
            obs[3 * j + 2] = normalize_distance(np.linalg.norm(delta))

            i += 1

        return obs
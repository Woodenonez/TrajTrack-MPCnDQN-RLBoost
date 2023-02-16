import numpy.typing as npt
import numpy as np
from math import atan2, cos, sin
from . import Component
from .utils import normalize_distance


class ReferencePathSampleObservation(Component):
    """
    Observes the distance and relative angle to some equally spaced upcoming
    points along the reference path
    """
    def __init__(self, num_samples: int, spacing: float, offset: float):
        """
        :param num_samples: The number of observed points
        :param spacing: The spacing between samples along the path
        :param offset: The distance along the path from the closest point to
                       the robot on the path to the first sample
        """
        self.num_samples = num_samples
        self.spacing = spacing
        self.offset = offset

        self.internal_obs_min = [-1, -1, 0] * self.num_samples
        self.internal_obs_max = [1, 1, 1] * self.num_samples

    def internal_obs(self) -> npt.ArrayLike:
        obs = np.zeros((3 * self.num_samples,), dtype=np.float32)
        for i in range(self.num_samples):
            point = self.env.path.interpolate(self.env.path_progress + i * self.spacing + self.offset)
            delta = np.asarray(point.coords[0]) - self.env.agent.position
            
            absulute_point_angle = atan2(delta[1], delta[0])
            relative_point_angle = absulute_point_angle - self.env.agent.angle

            obs[3 * i + 0] = cos(relative_point_angle)
            obs[3 * i + 1] = sin(relative_point_angle)
            obs[3 * i + 2] = normalize_distance(np.linalg.norm(delta))

        return obs
import numpy.typing as npt
from numpy.linalg import norm
from . import Component
from .utils import normalize_distance


class GoalDistanceObservation(Component):
    """
    Observes the distance from the robot to the goal position
    """
    internal_obs_min: npt.ArrayLike = [0]
    internal_obs_max: npt.ArrayLike = [1]

    def internal_obs(self) -> npt.ArrayLike:
        distance = norm(self.env.goal.position - self.env.agent.position)
        return [normalize_distance(distance)]
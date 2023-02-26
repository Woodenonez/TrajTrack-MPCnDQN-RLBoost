import numpy.typing as npt
from math import cos, sin, atan2
from . import Component


class GoalAngleObservation(Component):
    """
    Observes the relative angle from the robot to the goal position
    """
    internal_obs_min: npt.ArrayLike = [-1, -1]
    internal_obs_max: npt.ArrayLike = [1, 1]

    def internal_obs(self) -> npt.ArrayLike:
        delta = self.env.goal.position - self.env.agent.position
        absulute_goal_angle = atan2(delta[1], delta[0])
        relative_goal_angle = absulute_goal_angle - self.env.agent.angle
        return [cos(relative_goal_angle), sin(relative_goal_angle)]
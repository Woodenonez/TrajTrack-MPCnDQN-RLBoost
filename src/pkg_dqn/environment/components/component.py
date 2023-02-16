import numpy.typing as npt
from matplotlib.axes import Axes
from gym import spaces
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..environment import TrajectoryPlannerEnvironment


class Component:
    """
    Base class for all components
    """
    env: 'TrajectoryPlannerEnvironment'

    internal_obs_min: npt.ArrayLike = []
    internal_obs_max: npt.ArrayLike = []

    external_obs_space: spaces.Box = spaces.Box(0, 0, shape=())

    def internal_obs(self) -> npt.ArrayLike:
        return []

    def external_obs(self) -> npt.ArrayLike:
        return []
    
    def reset(self) -> None:
        pass

    def step(self, action: int) -> float:
        reward = 0
        return reward

    def render(self, ax: Axes) -> None:
        pass
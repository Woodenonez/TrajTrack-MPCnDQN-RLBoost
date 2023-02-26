import numpy.typing as npt
from . import Component
from .utils import normalize


class SpeedObservation(Component):
    """
    Observes robot speed
    """
    internal_obs_min: npt.ArrayLike = [-1]
    internal_obs_max: npt.ArrayLike = [1]

    def internal_obs(self) -> npt.ArrayLike:
        return [normalize(self.env.agent.speed, self.env.agent.cfg.SPEED_MIN, self.env.agent.cfg.SPEED_MAX)]
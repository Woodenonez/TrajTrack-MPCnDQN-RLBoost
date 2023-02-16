from . import Component


class CrossTrackReward(Component):
    """
    Gives a reward proportional to -(path cross track error)²
    """
    def __init__(self, factor: float):
        self.factor = factor
    
    def step(self, action: int) -> float:
        closest_point = self.env.path.interpolate(self.env.path_progress)
        cte = self.env.agent.point.distance(closest_point)
        return -self.env.time_step * self.factor * cte**2
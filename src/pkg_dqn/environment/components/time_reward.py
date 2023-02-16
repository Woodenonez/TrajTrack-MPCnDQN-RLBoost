from . import Component


class TimeReward(Component):
    """
    Gives a constant negative reward at each time step
    """
    def __init__(self, factor: float):
        self.factor = factor
    
    def step(self, action: int) -> float:
        reward = -self.factor * self.env.time_step
        return reward
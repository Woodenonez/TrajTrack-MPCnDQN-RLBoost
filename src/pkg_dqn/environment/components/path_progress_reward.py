from . import Component


class PathProgressReward(Component):
    """
    Rewards making progress along the reference path, penalizes the opposite
    """
    def __init__(self, factor: float):
        self.factor = factor

    def reset(self) -> None:
        self.last_path_progress = 0
    
    def step(self, action: int) -> float:
        reward = self.factor * (self.env.path_progress - self.last_path_progress)
        self.last_path_progress = self.env.path_progress
        return reward
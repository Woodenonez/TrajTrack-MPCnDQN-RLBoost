from . import Component


class ReachGoalReward(Component):
    """
    Gives a constant negative reward when the robot reaches the goal
    """
    def __init__(self, factor: float):
        self.factor = factor
    
    def step(self, action: int) -> float:
        reward = self.factor if self.env.reached_goal else 0
        return reward
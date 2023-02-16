from numpy.linalg import norm
from . import Component


class GoalDistanceReward(Component):
    """
    Rewards decreasing the distance to the goal and penalizes increasing said
    distance
    """
    def __init__(self, factor: float):
        self.factor = factor

    def reset(self) -> None:
        self.distance_to_goal = norm(self.env.goal.position - self.env.agent.position)

    def step(self, action: int) -> float:
        self.last_distance_to_goal = self.distance_to_goal
        self.distance_to_goal = norm(self.env.goal.position - self.env.agent.position)
        if self.last_distance_to_goal is None:
            self.distance_to_goal = self.last_distance_to_goal
        
        reward = self.factor * (self.last_distance_to_goal - self.distance_to_goal)
        return reward

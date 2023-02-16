from . import Component


class CollisionReward(Component):
    """
    Gives a constant negative reward when the robot collides with something
    """
    def __init__(self, factor: float):
        self.factor = factor
    
    def step(self, action: int) -> float:
        reward = -self.factor if self.env.collided else 0
        return reward
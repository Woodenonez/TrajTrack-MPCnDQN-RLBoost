from .agent import MobileRobot
from .obstacle import Boundary, Obstacle, Animation
from .goal import Goal

from typing import Callable, Tuple, List

MapDescription = Tuple[MobileRobot, Boundary, List[Obstacle], Goal]
MapGenerator = Callable[[], MapDescription]

__all__ = ['MobileRobot', 'Boundary', 'Obstacle', 'Animation', 'Goal', 'MapDescription']

from gym.envs.registration import register
# from gym import register

max_episode_steps = 1000
register(
    id='TrajectoryPlannerEnvironmentImgsReward1-v0',
    entry_point='pkg_dqn.environment.variants.imgs_reward1:TrajectoryPlannerEnvironmentImgsReward1',
    max_episode_steps=max_episode_steps,
)
register(
    id='TrajectoryPlannerEnvironmentRaysReward1-v0',
    entry_point='pkg_dqn.environment.variants.rays_reward1:TrajectoryPlannerEnvironmentRaysReward1',
    max_episode_steps=max_episode_steps,
)

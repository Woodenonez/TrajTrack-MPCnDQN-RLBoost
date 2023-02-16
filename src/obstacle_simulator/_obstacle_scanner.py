from abc import ABC, abstractmethod

from ._obstacle_simulator import ObstacleSimulator

from typing import List, Union, Tuple


class ObstacleScanner(ABC):
    """This is for multiple obstacles, for only one obstacle, use ObstacleSimulator directly."""
    def __init__(self, obstacle_sims: List[ObstacleSimulator]):
        self.obstacle_sims = obstacle_sims

    def get_full_obstacle_list(self, current_time, factor) -> list:
        """Get the list of all modes of all obstacles at the current time.
        Each mode is regarded as a separate obstacle."""
        all_obstacle_list = []
        for obs_sim in self.obstacle_sims:
            obstacle_modes = obs_sim.get_full_obstacle_list(current_time, factor)
            all_obstacle_list.extend(obstacle_modes)
        return all_obstacle_list
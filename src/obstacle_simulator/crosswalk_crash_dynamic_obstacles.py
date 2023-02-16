import os
import numpy as np

from ._obstacle_simulator import ObstacleSimulator

from typing import Union


class CrosswalkCrashObstacleSimulator(ObstacleSimulator):
    def __init__(self, sampling_time: float, radius: float, birth_time: float, prediction_time_offset:int=20, speed:float=1.5) -> None:
        super().__init__(sampling_time, radius, birth_time, prediction_time_offset)
        self.num_mode = 1
        self._load_preset()
        self.speed = speed
        self._create_obstacle()

    def _load_preset(self):
        self.y_coord = 3.5 # the y-coord of the obstacle (default: 3.5)
        self.sigma_x = 0.5 # the uncertainty of the obstacle in x-direction (default: 0.5)
        self.sigma_y = 0.5 # the uncertainty of the obstacle in y-direction (default: 0.5)

    def _create_obstacle(self):
        speed_per_step = self.ts * self.speed

        self.obj_list = []

        x_ = np.arange(start=16, stop=0, step=-speed_per_step).tolist()
        y_ = [self.y_coord] * len(x_)
        angle = np.pi/2

        for k in range(0, len(x_)):
            this_dict = {'info':[k, x_[k], y_[k]]}
            for i, key in enumerate([f'pred_T{i}' for i in range(1,self.T_max+1)]):
                if (k+1+i) < len(x_):
                    this_dict[key] = [[1, x_[k+i+1], y_[k+i+1], self.sigma_x, self.sigma_y, angle]]
                else:
                    this_dict[key] = [[1, x_[-1], y_[-1], self.sigma_x, self.sigma_y, angle]]
            self.obj_list.append(this_dict)
        return self.obj_list




        

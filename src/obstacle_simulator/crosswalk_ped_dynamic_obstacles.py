import os
import numpy as np

from ._obstacle_simulator import ObstacleSimulator

from typing import Union


class CrosswalkPedObstacleSimulator(ObstacleSimulator):
    def __init__(self, sampling_time: float, radius: float, birth_time: float, mode:int=2, prediction_time_offset:int=20, speed:float=1.2) -> None:
        super().__init__(sampling_time, radius, birth_time, prediction_time_offset)
        if (mode not in [1,2]):
            raise ValueError(f'Mode {mode} not defined.')
        self.num_mode = 2
        self._load_preset()
        self.speed = speed
        self._create_obstacle(mode=mode)

    def _load_preset(self):
        self.sigma_x = 0.2 # the uncertainty of the obstacle in x-direction (default: 0.2)
        self.sigma_y = 0.2 # the uncertainty of the obstacle in y-direction (default: 0.2)
        self.alpha_1 = 0.5 # probability for not crossing
        self.alpha_2 = 0.5 # probability for crossing

    def _create_obstacle(self, mode:int=2):
        speed_per_step = self.ts * self.speed
        
        self.obj_list = []

        x_before_crossing = np.arange(start=0, stop=10, step=speed_per_step).tolist()
        y_before_crossing = [0.8] * len(x_before_crossing)
        x_m1 = np.arange(start=10+speed_per_step, stop=16, step=speed_per_step).tolist() # no crossing
        y_m1 = y_before_crossing + [0.8] * len(x_m1) # no crossing
        x_m1 = x_before_crossing + x_m1
        y_m2 = np.arange(start=0.8+speed_per_step, stop=9, step=speed_per_step).tolist() # do crossing
        x_m2 = x_before_crossing + [10] * len(y_m2) # do crossing
        y_m2 = y_before_crossing + y_m2
        len_diff = abs(len(x_m1)-len(x_m2))
        if len(x_m1)>len(x_m2):
            x_m2 += [x_m2[-1]]*len_diff
            y_m2 += [y_m2[-1]]*len_diff
        else:
            x_m1 += [x_m1[-1]]*len_diff
            y_m1 += [y_m1[-1]]*len_diff
        x_m1 += [x_m1[-1]]*100
        y_m1 += [y_m1[-1]]*100
        x_m2 += [x_m2[-1]]*100
        y_m2 += [y_m2[-1]]*100

        angle_0 = 0
        if mode == 1:
            angle_1 = 0
        else:
            angle_1 = np.pi/4

        this_x = [x_m1, x_m2][mode-1]
        this_y = [y_m1, y_m2][mode-1]
        for k in range(0, len(this_x)-1-self.T_max):
            this_dict = {'info':[k, this_x[k], this_y[k]]}
            if k < len(x_before_crossing): # with two potential modes [k < len(x_before_crossing)]
                for i, key in enumerate([f'pred_T{x}' for x in range(1,self.T_max+1)]):
                    this_dict[key] = [[self.alpha_1, x_m1[k+i+1], y_m1[k+i+1], self.sigma_x*(i+1)/self.T_max, self.sigma_y*(i+1)/self.T_max, angle_0], 
                                      [self.alpha_2, x_m2[k+i+1], y_m2[k+i+1], self.sigma_x*(i+1)/self.T_max, self.sigma_y*(i+1)/self.T_max, angle_0]]
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,self.T_max+1)]):
                    this_dict[key] = [[1, this_x[k+i+1], this_y[k+i+1], self.sigma_x*(i+1)/self.T_max, self.sigma_y*(i+1)/self.T_max, angle_1]]
            self.obj_list.append(this_dict)
        return self.obj_list

    



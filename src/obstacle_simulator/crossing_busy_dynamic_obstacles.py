import os
import numpy as np
import pandas as pd

from ._obstacle_simulator import ObstacleSimulator
from ._obstacle_scanner import ObstacleScanner

from typing import Tuple, List, Union

def move_agent(start: np.ndarray, goal: np.ndarray, ts: float, speed: float) -> Tuple[list, list]:
    speed_per_step = ts * speed
    distance = np.linalg.norm((goal-start))
    total_time_steps = int(distance / speed_per_step)
    x_coord = np.linspace(start=start[0], stop=goal[0], num=total_time_steps)
    y_coord = np.linspace(start=start[1], stop=goal[1], num=total_time_steps)
    return x_coord.tolist(), y_coord.tolist()

def move_agent_waypoints(waypoints:np.ndarray,  ts: float, speed:float) -> Tuple[list, list]:
    """
    Args:
        waypoints: n*2, every row is a pair of waypoint coordinates
    """
    x_coord = []
    y_coord = []
    for i in range(waypoints.shape[0]-1):
        x_part, y_part = move_agent(start=waypoints[i,:], goal=waypoints[i+1,:], ts=ts, speed=speed)
        x_coord += x_part
        y_coord += y_part
    return x_coord, y_coord


class CrossingObstacleScanner(ObstacleScanner):
    def __init__(self, sampling_time: float, obstacle_radius_list: List[float]):
        if len(obstacle_radius_list) != 5:
            raise ValueError("The length of the obstacle radius list should be 5.")
        obs_ped_1 = _ObstacleSimulator_Ped_1(sampling_time, obstacle_radius_list[0])
        obs_ped_2 = _ObstacleSimulator_Ped_2(sampling_time, obstacle_radius_list[1])
        obs_ped_3 = _ObstacleSimulator_Ped_3(sampling_time, obstacle_radius_list[2])
        obs_veh_1 = _ObstacleSimulator_Veh_1(sampling_time, obstacle_radius_list[3])
        obs_veh_2 = _ObstacleSimulator_Veh_2(sampling_time, obstacle_radius_list[4])
        super().__init__([obs_ped_1, obs_ped_2, obs_ped_3, obs_veh_1, obs_veh_2])


class _ObstacleSimulator_Ped_1(ObstacleSimulator):
    def __init__(self, sampling_time: float, radius: float, birth_time:float=-1.4, mode:int=1, prediction_time_offset:int=20, speed:float=1.0) -> None:
        super().__init__(sampling_time, radius, birth_time, prediction_time_offset)
        if (mode not in [1,2]): # 1-crossing; 2-avoiding
            raise ValueError(f'Mode {mode} not defined.')
        self.num_mode = 3
        self._load_preset()
        self.speed = speed
        self._create_obstacle(mode)


    def _load_preset(self):
        self.sigma_x = 0.2
        self.sigma_y = 0.2
        self.alpha_1 = 0.5 # probability for not crossing
        self.alpha_2 = 0.5 # probability for crossing

    def _create_obstacle(self, mode:int=1):
        T = self.T_max
        a1, a2, sx, sy = self.alpha_1, self.alpha_2, self.sigma_x, self.sigma_y
        self.obj_list = []

        x_, y_   = move_agent(np.array([12,  3.5]), np.array([8.5, 3.5]), self.ts, self.speed)
        x_1, y_1 = move_agent(np.array([8.5, 3.5]), np.array([0,   3.5]), self.ts, self.speed)
        x_2, y_2 = move_agent(np.array([8.5, 3.5]), np.array([8.5, 12]),  self.ts, self.speed)
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2
        y_m2 = y_ + y_2

        angle = 0

        x_m_list = [x_m1, x_m2]
        y_m_list = [y_m1, y_m2]
        this_x = x_m_list[mode-1]
        this_y = y_m_list[mode-1]
        for k in range(0, len(this_x)-1):
            this_dict = {'info':[k, this_x[k], this_y[k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], x_m_list, y_m_list):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sx*(i+1)/T, sy*(i+1)/T, angle])
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, this_x[k+i+1], this_y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle]]
                    except:
                        this_dict[key] = [[1, this_x[-1],    this_y[-1],    sx*(i+1)/T, sy*(i+1)/T, angle]]
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list

class _ObstacleSimulator_Ped_2(ObstacleSimulator):
    def __init__(self, sampling_time: float, radius: float, birth_time:float=-2.4, mode:int=1, prediction_time_offset:int=20, speed:float=1.0) -> None:
        super().__init__(sampling_time, radius, birth_time, prediction_time_offset)
        if (mode not in [1,2,3]): # 1,3-crossing; 2-avoiding
            raise ValueError(f'Mode {mode} not defined.')
        self.num_mode = 3
        self._load_preset()
        self.speed = speed
        self._create_obstacle(mode)


    def _load_preset(self):
        self.sigma_x = 0.2
        self.sigma_y = 0.2
        self.alpha_1 = 0.5 # probability for not crossing
        self.alpha_2 = 0.5 # probability for crossing

    def _create_obstacle(self, mode:int=1):
        T = self.T_max
        a1, a2, sx, sy = self.alpha_1, self.alpha_2, self.sigma_x, self.sigma_y
        self.obj_list = []

        x_, y_   = move_agent(np.array([8.5, 0]),   np.array([8.5, 3.6]), self.ts, self.speed)
        x_1, y_1 = move_agent(np.array([8.5, 3.6]), np.array([0,   3.6]), self.ts, self.speed)
        x_2, y_2 = move_agent(np.array([8.5, 3.6]), np.array([8.5, 8.5]), self.ts, self.speed)
        x_2_1, y_2_1 = move_agent(np.array([8.5, 8.5]), np.array([8.5, 12]), self.ts, self.speed)
        x_2_2, y_2_2 = move_agent(np.array([8.5, 8.5]), np.array([0, 8.5]),  self.ts, self.speed)
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2 + x_2_1
        y_m2 = y_ + y_2 + y_2_1
        x_m3 = x_ + x_2 + x_2_2
        y_m3 = y_ + y_2 + y_2_2

        angle = 0

        x_m_list = [x_m1, x_m2, x_m3]
        y_m_list = [y_m1, y_m2, y_m3]

        this_x = x_m_list[mode-1]
        this_y = y_m_list[mode-1]
        for k in range(0, len(this_x)-1):
            this_dict = {'info':[k, this_x[k], this_y[k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m1, x_m2], [y_m1, y_m2]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sx*(2)*(i+1)/T, sy*(2)*(i+1)/T, angle])
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            elif (mode!=1) and (k < (len(x_)+len(x_2))):
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m2, x_m3], [y_m2, y_m3]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sx*(i+1)/T, sy*(i+1)/T, angle])
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, this_x[k+i+1], this_y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle]]
                    except:
                        this_dict[key] = [[1, this_x[-1],    this_y[-1],    sx*(i+1)/T, sy*(i+1)/T, angle]]
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list

class _ObstacleSimulator_Ped_3(ObstacleSimulator):
    def __init__(self, sampling_time: float, radius: float, birth_time:float=4.0, mode:int=1, prediction_time_offset:int=20, speed:float=1.0) -> None:
        super().__init__(sampling_time, radius, birth_time, prediction_time_offset)
        if (mode not in [1,2,3]): # 1-crossing;   2,3-avoiding
            raise ValueError(f'Mode {mode} not defined.')
        self.num_mode = 3
        self._load_preset()
        self.speed = speed
        self._create_obstacle(mode)

    def _load_preset(self):
        self.sigma_x = 0.2
        self.sigma_y = 0.2
        self.alpha_1 = 0.4 # do crossing
        self.alpha_2 = 0.3 # no crossing
        self.alpha_3 = 0.3 # no crossing

    def _create_obstacle(self, mode:int=1):
        T = self.T_max
        a1, a2, a3, sx, sy = self.alpha_1, self.alpha_2, self.alpha_3, self.sigma_x, self.sigma_y
        self.obj_list = []

        x_, y_   = move_agent(np.array([12,  8.5]), np.array([8.3, 8.5]), self.ts, self.speed)
        x_1, y_1 = move_agent(np.array([8.3, 8.5]), np.array([0, 8.5]),   self.ts, self.speed)
        x_2, y_2 = move_agent(np.array([8.3, 8.5]), np.array([8.3, 12]),  self.ts, self.speed)
        x_3, y_3 = move_agent(np.array([8.3, 8.5]), np.array([8.3, 0]),   self.ts, self.speed)
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2
        y_m2 = y_ + y_2
        x_m3 = x_ + x_3
        y_m3 = y_ + y_3

        angle = 0

        x_m_list = [x_m1, x_m2, x_m3]
        y_m_list = [y_m1, y_m2, y_m3]

        this_x = x_m_list[mode-1]
        this_y = y_m_list[mode-1]
        for k in range(0, len(this_x)-1):
            this_dict = {'info':[k, this_x[k], this_y[k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2, a3], x_m_list, y_m_list):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sx*(i+1)/T, sy*(i+1)/T, angle])
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, this_x[k+i+1], this_y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle]]
                    except:
                        this_dict[key] = [[1, this_x[-1],    this_y[-1],    sx*(i+1)/T, sy*(i+1)/T, angle]]
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list

class _ObstacleSimulator_Veh_1(ObstacleSimulator):
    def __init__(self, sampling_time: float, radius: float, birth_time:float=2.0, mode:int=2, prediction_time_offset:int=20, speed:float=1.0) -> None:
        super().__init__(sampling_time, radius, birth_time, prediction_time_offset)
        if (mode not in [1,2,3]): # 1-crossing;   2,3-avoiding
            raise ValueError(f'Mode {mode} not defined.')
        self.num_mode = 3
        self._load_preset()
        self.speed = speed
        self._create_obstacle(mode)

    def _load_preset(self):
        self.sigma_x = 0.4
        self.sigma_y = 0.4
        self.alpha_1 = 0.5 # do crossing
        self.alpha_2 = 0.5 # no crossing
        # a3 = 0

    def _create_obstacle(self, mode:int=1):
        T = self.T_max
        a1, a2, sx, sy = self.alpha_1, self.alpha_2, self.sigma_x, self.sigma_y
        self.obj_list = []

        x_, y_   = move_agent(np.array([12, 7]), np.array([9, 7]), self.ts, self.speed)
        x_1, y_1 = move_agent_waypoints(np.array([[9, 7], [7, 7], [7, 12]]), self.ts, self.speed) # turn right
        x_2, y_2 = move_agent(np.array([9, 7]), np.array([5, 7]), self.ts, self.speed) # go straight
        x_2_1, y_2_1 = move_agent(np.array([5, 7]), np.array([5, 0]), self.ts, self.speed) # turn left
        x_2_2, y_2_2 = move_agent(np.array([5, 7]), np.array([0, 7]), self.ts, self.speed) # go straight
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2 + x_2_1
        y_m2 = y_ + y_2 + y_2_1
        x_m3 = x_ + x_2 + x_2_2
        y_m3 = y_ + y_2 + y_2_2

        angle = 0

        x_m_list = [x_m1, x_m2, x_m3]
        y_m_list = [y_m1, y_m2, y_m3]
        this_x = x_m_list[mode-1]
        this_y = y_m_list[mode-1]
        for k in range(0, len(this_x)-1):
            this_dict = {'info':[k, this_x[k], this_y[k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m1, x_m2], [y_m1, y_m2]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sx*(i+1)/T, sy*(i+1)/T, angle])
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            elif (mode!=1) and (k < (len(x_)+len(x_2))):
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m2, x_m3], [y_m2, y_m3]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sx*(i+1)/T, sy*(i+1)/T, angle])
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, this_x[k+i+1], this_y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle]]
                    except:
                        this_dict[key] = [[1, this_x[-1],    this_y[-1],    sx*(i+1)/T, sy*(i+1)/T, angle]]
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list

class _ObstacleSimulator_Veh_2(ObstacleSimulator):
    def __init__(self, sampling_time: float, radius: float, birth_time:float=-1.0, mode:int=2, prediction_time_offset:int=20, speed:float=1.0) -> None:
        super().__init__(sampling_time, radius, birth_time, prediction_time_offset)
        if (mode not in [1,2,3]): # 1-crossing;   2,3-avoiding
            raise ValueError(f'Mode {mode} not defined.')
        self.num_mode = 3
        self._load_preset()
        self.speed = speed
        self._create_obstacle(mode)

    def _load_preset(self):
        self.sigma_x = 0.4
        self.sigma_y = 0.4
        self.alpha_1 = 0.5 # do crossing
        self.alpha_2 = 0.5 # no crossing
        # a3 = 0

    def _create_obstacle(self, mode:int=1):
        T = self.T_max
        a1, a2, sx, sy = self.alpha_1, self.alpha_2, self.sigma_x, self.sigma_y
        self.obj_list = []

        x_, y_   = move_agent(np.array([0, 5]), np.array([3, 5]), self.ts, self.speed)
        x_1, y_1 = move_agent_waypoints(np.array([[3, 5], [5, 5], [5, 0]]),  self.ts, self.speed) # turn right
        x_2, y_2 = move_agent(np.array([3, 5]), np.array([7, 5]),  self.ts, self.speed)
        x_2_1, y_2_1 = move_agent(np.array([7, 5]), np.array([7, 12]),  self.ts, self.speed) # turn left
        x_2_2, y_2_2 = move_agent(np.array([7, 5]), np.array([12, 5]),  self.ts, self.speed) # go straight
        x_m1 = x_ + x_1
        y_m1 = y_ + y_1
        x_m2 = x_ + x_2 + x_2_1
        y_m2 = y_ + y_2 + y_2_1
        x_m3 = x_ + x_2 + x_2_2
        y_m3 = y_ + y_2 + y_2_2

        angle = 0

        x_m_list = [x_m1, x_m2, x_m3]
        y_m_list = [y_m1, y_m2, y_m3]
        this_x = x_m_list[mode-1]
        this_y = y_m_list[mode-1]
        for k in range(0, len(this_x)-1):
            this_dict = {'info':[k, this_x[k], this_y[k]]}
            if k < len(x_): # with two potential modes
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m1, x_m2], [y_m1, y_m2]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sx*(i+1)/T, sy*(i+1)/T, angle])
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            elif (mode!=1) and (k < (len(x_)+len(x_2))):
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    this_dict[key] = []
                    for a, x, y in zip([a1, a2], [x_m2, x_m3], [y_m2, y_m3]):
                        try:
                            this_dict[key].append([a, x[k+i+1], y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle])
                        except:
                            this_dict[key].append([a, x[-1], y[-1], sx*(i+1)/T, sy*(i+1)/T, angle])
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            else: # collapse to one mode
                for i, key in enumerate([f'pred_T{x}' for x in range(1,T+1)]):
                    try:
                        this_dict[key] = [[1, this_x[k+i+1], this_y[k+i+1], sx*(i+1)/T, sy*(i+1)/T, angle]]
                    except:
                        this_dict[key] = [[1, this_x[-1],    this_y[-1],    sx*(i+1)/T, sy*(i+1)/T, angle]]
                    this_dict[key] += [[0, 0, 0, 1, 1, 0]] * (self.num_mode-len(this_dict[key])) # number of modes are fixed
            self.obj_list.append(this_dict)
        return self.obj_list


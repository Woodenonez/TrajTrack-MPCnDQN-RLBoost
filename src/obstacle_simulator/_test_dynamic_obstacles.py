import os, sys
import numpy as np

try:
    from mpc_planner.test_maps.test_graphs import Graph
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from map_generator.test_graphs import Graph

'''
File info:
    Name    - [test_dynamic_obstacles]
    Exe     - [No]
File description:
    It is used to generate the dynamic obstacles for testing.
File content:
    ObstacleScanner <class> - Offering information and MPC-format list of dynamic obstacles.
'''

class ObstacleScanner():
    '''
    Description:
        Generate/read the information of dynamic obstacles, and form it into the correct format for MPC.
    Attributes:
        num_obstacles <int> - The number of (active) dynamic obstacles.
    Functions
        get_obstacle_info      <get> - Get the position, orientation, and shape information of a selected (idx) obstacle.
        get_full_obstacle_list <get> - Form the list for MPC problem.
    Comments:
        Other attributes and functions are just for testing and specific in thie file.
    '''
    def __init__(self, graph:Graph):
        self.__dyn_obs_list = graph.dyn_obs_list # x, y, freq, rx, ry, theta, alpha
        self.num_obstacles = len(self.__dyn_obs_list)

    def get_obstacle_info(self, idx, current_time, key):
        obstacle = self.__dyn_obs_list[idx]
        p1, p2, freq, rx, ry, angle = obstacle
        pos = self.gen_obstacle_point(p1, p2, freq, current_time)
        obs_dict = {'position':tuple(pos), 'pos':tuple(pos), 'radius':(rx,ry), 'axis':(rx,ry), 'heading':angle, 'angle':angle}
        return obs_dict[key]

    def get_full_obstacle_list(self, current_time, horizon, ts=0.2):
        obstacle_list = self.get_dyn_obstacle(current_time, horizon, ts)
        return obstacle_list # x, y, rx, ry, theta, alpha for t in horizon

    # Below just for testing
    def gen_obstacle_point(self, p1, p2, freq, time):
        time = np.array(time)
        t = abs(np.sin(freq * time))
        if type(t) == np.ndarray:
            t = np.expand_dims(t,1)
        p3 = t*np.array(p1) + (1-t)*np.array(p2)
        return p3

    def get_dyn_obstacle(self, t, horizon, ts):
        vehicle_width = 0.5
        vehicle_margin = 0.25
        if len(self.__dyn_obs_list) == 0:
            return []
        time = np.linspace(t, t+horizon*ts, horizon)
        obs_list = []
        for obs in self.__dyn_obs_list:
            p1, p2, freq, x_radius, y_radius, angle = obs
            x_radius = x_radius+vehicle_width/2+vehicle_margin
            y_radius = y_radius+vehicle_width/2+vehicle_margin

            q = [(*self.gen_obstacle_point(p1, p2, freq, t), x_radius, y_radius, angle, 1) for t in time] # alpha=1
            obs_list.append(q)
        return obs_list # x, y, rx, ry, theta, alpha

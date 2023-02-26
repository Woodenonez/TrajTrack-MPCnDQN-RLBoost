import math
import random
import statistics

import numpy as np
from shapely.geometry import Polygon, Point, LineString

from pkg_map.map_geometric import GeometricMap
from pkg_obstacle import geometry_tools

from pkg_dqn.environment import MapDescription
from pkg_dqn.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc
from pkg_dqn.utils.map import generate_map_scene_1, generate_map_scene_2

from typing import List, Tuple


class Inflator:
    def __init__(self, inflate_margin):
        self.inflate_margin = inflate_margin

    def __call__(self, polygon: List[tuple]):
        shapely_inflated = geometry_tools.polygon_inflate(Polygon(polygon), self.inflate_margin)
        return geometry_tools.polygon_to_vertices(shapely_inflated)
    

class HintSwitcher:
    def __init__(self, max_switch_distance: float, min_detach_distance: float, min_detach_steps:float=5):
        self.switch_distance = max_switch_distance
        self.detach_distance = min_detach_distance
        self.detach_steps = min_detach_steps
        self.detach_cnt = 0
        self.switch_on = False

    def switch(self, current_position: tuple, original_traj: List[tuple], new_traj: List[tuple], obstacle_list: List[List[tuple]]) -> bool:
        cnt_flag = False
        for old_pos, new_pos in zip(original_traj, new_traj):
            for obstacle in obstacle_list:
                shapely_obstacle = Polygon(obstacle)
                dist = shapely_obstacle.distance(Point(current_position))
                if shapely_obstacle.contains(Point(old_pos[:2])): # or shapely_obstacle.contains(Point(new_pos[:2])):
                    if (dist < self.switch_distance) & (self.switch_on == False):
                            self.switch_on = True
                            return self.switch_on
                elif (dist > self.detach_distance) & (self.switch_on == True):
                    if self.detach_cnt > self.detach_steps:
                        self.switch_on = False
                        self.detach_cnt = 0
                    elif cnt_flag == False:
                        self.detach_cnt += 1
                        cnt_flag = True
        return self.switch_on


class Metrics:
    """
    Metrics for the environment.
    1. Computation time (Average, Max, Median for one run)
    2. Deviation distance (Average, Max, for one run)
    3. Action smoothness (https://www.mathworks.com/help/nav/ref/pathmetrics.smoothness.html) (second order derivative)
    4. Minimal clearance / obstacle distance (in one run)
    5. Finish time (in one run)
    6. Success rate (in 10 runs)
    """
    def __init__(self, mode: str) -> None:
        """
        Args:
            mode: dqn, mpc, or hyb
        """
        if mode not in ['dqn', 'mpc', 'hyb']:
            raise ValueError(f"Mode {mode} not recognized (should be 'dqn', 'mpc', or 'hyb').")
        self.mode = mode
        self.trial_list = []
        self.success_rate = 0

    def get_average(self, round_digit:int=4) -> dict:
        self.metric_average = {}
        all_compuation_time = []
        all_deviation_distance = []
        all_smoothness = []
        all_clearance = []
        all_finish_time = []
        for trial in self.trial_list:
            trial: dict
            all_compuation_time.append(trial["computation_time"])
            all_deviation_distance.append(trial["deviation_distance"])
            all_smoothness.append(trial["smoothness"])
            all_clearance.append(trial["clearance"])
            if trial["success"]:
                all_finish_time.append(trial["finish_time"])
        if not all_finish_time:
            all_finish_time = [-1]
        self.metric_average["computation_time"] = [round(statistics.mean([x[0] for x in all_compuation_time]), round_digit),
                                                   round(statistics.mean([x[1] for x in all_compuation_time]), round_digit),
                                                   round(statistics.mean([x[2] for x in all_compuation_time]), round_digit)]
        self.metric_average["deviation_distance"] = [round(statistics.mean([x[0] for x in all_deviation_distance]), round_digit),
                                                     round(statistics.mean([x[1] for x in all_deviation_distance]), round_digit)]
        self.metric_average["smoothness"] = [round(statistics.mean([x[0] for x in all_smoothness]), round_digit),
                                             round(statistics.mean([x[1] for x in all_smoothness]), round_digit)]
        self.metric_average["clearance"] = round(statistics.mean(all_clearance), round_digit)
        self.metric_average["finish_time"] = round(statistics.mean(all_finish_time), round_digit)
        self.metric_average["success_rate"] = self.success_rate
        return self.metric_average

    def add_trial_result(self, computation_time_list: List[float], succeed: bool, action_list: List[tuple],
                         ref_trajectory: List[tuple], actual_trajectory: List[tuple], obstacle_list: List[List[tuple]]):
        metric_dict = {}
        metric_dict["computation_time"] = self._get_computation_time(computation_time_list)
        metric_dict["deviation_distance"] = self._get_deviation_distance(ref_trajectory, actual_trajectory)
        metric_dict["smoothness"] = self._get_smoothness(action_list)
        metric_dict["clearance"] = self._get_minimal_obstacle_distance(actual_trajectory, obstacle_list)
        metric_dict["finish_time"] = self._get_finish_time_steps(computation_time_list, succeed=succeed)
        metric_dict["success"] = True if metric_dict["finish_time"]>0 else False
        self.trial_list.append(metric_dict)
        self._get_success_rate()

    def _get_computation_time(self, computation_time_list):
        return [statistics.mean(computation_time_list), max(computation_time_list), statistics.median(computation_time_list)]
    
    def _get_deviation_distance(self, ref_traj: List[tuple], actual_traj: List[tuple]):
        deviation_dists = []
        for pos in actual_traj:
            deviation_dists.append(min([math.hypot(ref_pos[0]-pos[0], ref_pos[1]-pos[1]) for ref_pos in ref_traj]))
        return [statistics.mean(deviation_dists), max(deviation_dists)]
    
    def _get_smoothness(self, action_list: List[tuple]): 
        speeds = np.array(action_list)[:, 0]
        angular_speeds = np.array(action_list)[:, 1]
        return [statistics.mean(np.abs(np.diff(speeds, n=2))), statistics.mean(np.abs(np.diff(angular_speeds, n=2)))]

    def _get_minimal_obstacle_distance(self, trajectory: List[tuple], obstacles: List[List[tuple]]):
        dist_list = []
        for pos in trajectory:
            dist_list.append(min([Polygon(obs).distance(Point(pos)) for obs in obstacles]))
        return min(dist_list)

    def _get_finish_time_steps(self, computation_time_list, succeed: bool):
        if succeed:
            return len(computation_time_list)
        else:
            return -1
        
    def _get_success_rate(self):
        self.success_rate = sum([trial["success"] for trial in self.trial_list])/len(self.trial_list)


def get_geometric_map(rl_map: MapDescription, inflate_margin: float) -> GeometricMap:
    _, rl_boundary, rl_obstacles, _ = rl_map
    inflator = Inflator(inflate_margin)
    geometric_map = GeometricMap(
        boundary_coords=rl_boundary.vertices.tolist(),
        obstacle_list=[obs.nodes.tolist() for obs in rl_obstacles if obs.is_static],
        inflator=inflator
    )
    return geometric_map

def generate_map(scene:int=1, sub_scene:int=1, sub_scene_option:int=1, generator:bool=True) -> MapDescription:
    """
    MapDescription = Tuple[MobileRobot, Boundary, List[Obstacle], Goal]
    """
    if scene == None: # training
        return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc()])
    
    # return generate_map_dynamic
    # return generate_map_corridor
    # return generate_map_mpc()
    
    if scene == 1:
        map_des = generate_map_scene_1(sub_scene, sub_scene_option)
    elif scene == 2:
        map_des = generate_map_scene_2(sub_scene, sub_scene_option)
    elif scene == 3:
        map_des = generate_map_mpc(11)()
    else:
        raise ValueError(f"Scene {scene} not recognized (should be 1, 2, or 3).")
    
    def _generate_map():
        return map_des

    if generator:
        return _generate_map
    else:
        return map_des
    
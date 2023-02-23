import math
import itertools
import functools

import numpy as np
from shapely.geometry import Polygon

from pkg_motion_model import motion_model
from pkg_map.map_geometric import GeometricMap
from obstacle_simulator._obstacle_scanner import ObstacleScanner
from mpc_traj_tracker.trajectory_generator import TrajectoryGenerator
from visualizer.mpc_plot import MpcPlotInLoop

from pkg_obstacle import geometry_tools

from util import utils_geo
from util.mpc_config import Configurator

from typing import List


class Inflator:
    def __init__(self, inflate_margin):
        self.inflate_margin = inflate_margin

    def __call__(self, polygon: List[tuple]):
        shapely_inflated = geometry_tools.polygon_inflate(Polygon(polygon), self.inflate_margin)
        return geometry_tools.polygon_to_vertices(shapely_inflated)


class Simulator:
    def __init__(self, config:Configurator, scene_index, inflate_margin, use_tcp=False, verbose=False) -> None:
        if scene_index is None:
            self.__hint()
            scene_index = int(input('Please select a simulation index:'))
        self.config = config
        self.idx = scene_index
        self.inflate_margin = inflate_margin
        self.use_tcp = use_tcp
        self.vb = verbose

        self.inflator = Inflator(self.inflate_margin)

        # Commonly used variables
        self.ts = self.config.ts
        self.ns = self.config.ns
        self.nu = self.config.nu
        self.N_hor = self.config.N_hor

        self.__intro()
        self.load_map_and_obstacles()

        self.robot_dict = {}
        self.plotter = MpcPlotInLoop(self.config)

    def __hint(self):
        print('='*30)
        print('Index 0 - Test cases.')
        print('Index 1 - Single object, crosswalk.')
        print('Index 2 - Multiple objects, road crossing.')
        print('Index 3 - Single objects, crashing.')
        print('Index 4 - Single objects, following.')
        print('Index 5 - Two robots, crashing.')

    def __intro(self):
        assert(self.idx in [0,1,2,3,4,5]),(f'Index {self.idx} not found!')
        self.__hint()
        print(f'[{self.idx}] is selected.')
        print('='*30)

    def load_map_and_obstacles(self, test_graph_index=5):
        if self.idx == 0:
            raise NotImplementedError # TODO: implement this
            from pkg_map.preset_maps.test_maps import return_test_map
            boundary_coords, obstacle_list, start, end = return_test_map(test_graph_index)
            self.graph = GeometricMap(boundary_coords, obstacle_list, inflator=self.inflator)
            self.start = start
            self.waypoints = [end]
            self.scanner = ObstacleScanner(self.graph)

        elif self.idx == 1:
            from pkg_map.preset_maps.scene_maps import return_crosswalk_map, plot_crosswalk_map
            from obstacle_simulator.crosswalk_ped_dynamic_obstacles import CrosswalkPedObstacleSimulator
            boundary_coords, obstacle_list, cross_area = return_crosswalk_map()
            self.graph = GeometricMap(boundary_coords, obstacle_list, inflator=self.inflator)
            self.graph.plot = plot_crosswalk_map
            self.start = [(0.6, 3.5, math.radians(0))]
            self.waypoints = [[(15.4, 3.5, math.radians(0))]]
            self.scanner = CrosswalkPedObstacleSimulator(self.ts, 0.2, birth_time=-1)

        elif self.idx == 2:
            from pkg_map.preset_maps.scene_maps import return_crossing_map, plot_crossing_map
            from obstacle_simulator.crossing_busy_dynamic_obstacles import CrossingObstacleScanner
            boundary_coords, obstacle_list, sidewalk_list, crossing_area = return_crossing_map()
            self.graph = GeometricMap(boundary_coords, obstacle_list, inflator=self.inflator)
            self.graph.plot = plot_crossing_map
            self.start = [(7, 0.6, math.radians(90))]
            self.waypoints = [[(7, 11.5, math.radians(90)), (7, 15.4, math.radians(90))]]
            self.scanner = CrossingObstacleScanner(self.ts, obstacle_radius_list=[0.2, 0.2, 0.2, 0.5, 0.5])

        elif self.idx == 3:
            from pkg_map.preset_maps.scene_maps import return_crosswalk_map, plot_crosswalk_map
            from obstacle_simulator.crosswalk_crash_dynamic_obstacles import CrosswalkCrashObstacleSimulator
            boundary_coords, obstacle_list, cross_area = return_crosswalk_map(False)
            self.graph = GeometricMap(boundary_coords, obstacle_list, inflator=self.inflator)
            self.graph.plot = functools.partial(plot_crosswalk_map, with_static_obs=False)
            self.start = [(0.6, 3.5, math.radians(0))]
            self.waypoints = [[(15.4, 3.5, math.radians(0))]]
            self.scanner = CrosswalkCrashObstacleSimulator(self.ts, 0.5, birth_time=0)

        elif self.idx == 4:
            from pkg_map.preset_maps.scene_maps import return_crosswalk_map, plot_crosswalk_map
            from obstacle_simulator.crosswalk_follow_dynamic_obstacles import CrosswalkFollowObstacleSimulator
            boundary_coords, obstacle_list, cross_area = return_crosswalk_map(False)
            self.graph = GeometricMap(boundary_coords, obstacle_list, inflator=self.inflator)
            self.graph.plot = functools.partial(plot_crosswalk_map, with_static_obs=False)
            self.start = [(0.6, 3.5, math.radians(0))]
            self.waypoints = [[(15.4, 3.5, math.radians(0))]]
            self.scanner = CrosswalkFollowObstacleSimulator(self.ts, 0.2, birth_time=-3)

        elif self.idx == 5:
            from pkg_map.preset_maps.scene_maps import return_crosswalk_map, plot_crosswalk_map
            from obstacle_simulator.crosswalk_crash_dynamic_obstacles import CrosswalkCrashObstacleSimulator
            boundary_coords, obstacle_list, cross_area = return_crosswalk_map(False)
            self.graph = GeometricMap(boundary_coords, obstacle_list, inflator=self.inflator)
            self.graph.plot = functools.partial(plot_crosswalk_map, with_static_obs=False)
            self.start = [(0.6, 4.0, math.radians(0)),
                          (0.6, 3.0, math.radians(0))]
            self.waypoints = [[(15.4, 3.0, math.radians(180))], 
                              [(15.4, 4.0, math.radians(180))]]
            self.scanner = CrosswalkCrashObstacleSimulator(self.ts, 0.2, birth_time=0)

        else:
            raise ModuleNotFoundError
        
        return self.graph, self.scanner

    def load_robot(self, robot_id, ref_path:list, start:np.ndarray, end:np.ndarray, mode:str='work', color='b'):
        if robot_id in list(self.robot_dict):
            raise ValueError(f'Robot {robot_id} exists!')
        traj_gen = TrajectoryGenerator(self.config, self.use_tcp, verbose=self.vb)
        traj_gen.load_robot_dynamics(motion_model=motion_model.unicycle_model)
        this_robot_dict = {'traj_gen': traj_gen, 'start': start, 'end': end, 
                           'ref_path': ref_path, 'mode': mode, 'color': color, 'done':False,
                           'pred_states':None,} # pred_states is changed over time
        self.robot_dict[robot_id] = this_robot_dict

    def set_obstacle_weights(self, robot_id, stc_weights, dyn_weights):
        if robot_id not in list(self.robot_dict):
            raise ValueError(f'Robot {robot_id} does not exist!')
        traj_gen:TrajectoryGenerator = self.robot_dict[robot_id]['traj_gen']
        traj_gen.set_obstacle_weights(stc_weights, dyn_weights)

    def get_other_robot_states(self, robot_id):
        idx = 0
        other_robot_states = [0] * self.ns * self.N_hor * self.config.Nother
        for id in list(self.robot_dict):
            if id != robot_id:
                pred_states:np.ndarray = self.robot_dict[id]['pred_states'] # every row is a state
                if pred_states is not None:
                    other_robot_states[idx : idx+self.ns*self.N_hor] = list(pred_states.reshape(-1))
                    idx += self.ns*self.N_hor
        return other_robot_states

    def run(self, map_manager: GeometricMap, obstacle_scanner: ObstacleScanner, plot_in_loop=False):
        """Run the simulation
        
        Args:
            map_manager: The map manager, with map info and `plot()` method
            obstacle_scanner: The obstacle scanner, with `get_full_obstacle_list()` method
        
        Returns:
            robot_dict: A dictionary of robot info, with keys:
                `traj_gen`: TrajectoryGenerator object
                `start`: start state
                `end`: end state
                `ref_path`: reference path
                `mode`: work mode
                `color`: color for plotting
                `done`: whether the robot has finished its task
                `pred_states`: predicted states
        """
        ### Prepare for the loop computing ###
        for r_id in list(self.robot_dict):
            start    = self.robot_dict[r_id]['start']
            end      = self.robot_dict[r_id]['end']
            mode     = self.robot_dict[r_id]['mode']
            ref_path = self.robot_dict[r_id]['ref_path']
            traj_gen:TrajectoryGenerator = self.robot_dict[r_id]['traj_gen']
            traj_gen.load_init_state(start, end)
            traj_gen.set_work_mode(mode)
            traj_gen.set_ref_trajectory(ref_path)
        stc_constraints = [0.0] * self.config.Nstcobs * self.config.nstcobs

        ### Start the loop ###
        kt = 0  # time step, from 0 (kt*ts is the actual time)
        all_terminated = False

        ### Plot in loop
        if plot_in_loop:
            self.plotter.plot_in_loop_pre(map_manager)
            for r_id in list(self.robot_dict):
                start    = self.robot_dict[r_id]['start']
                end      = self.robot_dict[r_id]['end']
                mode     = self.robot_dict[r_id]['mode']
                ref_path = self.robot_dict[r_id]['ref_path']
                color    = self.robot_dict[r_id]['color']
                traj_gen:TrajectoryGenerator = self.robot_dict[r_id]['traj_gen']
                self.plotter.add_object_to_pre(r_id, traj_gen.ref_traj.numpy(), start, end, color=color)

        while (not all_terminated):
            ### Static obstacles
            map_boundry, map_obstacle_list = map_manager()
            for i, map_obstacle in enumerate(map_obstacle_list):
                b, a0, a1 = utils_geo.polygon_halfspace_representation(np.array(map_obstacle))
                stc_constraints[i*self.config.nstcobs : (i+1)*self.config.nstcobs] = (b+a0+a1)

            ### Dynamic obstacles [[(x, y, rx ,ry, angle, alpha),(),...],[(),(),...]] each sub-list is a mode_/obstacle  
            params_per_dyn_obs  = self.N_hor * self.config.ndynobs      
            dyn_constraints = [0.0] * self.config.Ndynobs * params_per_dyn_obs
            full_dyn_obstacle_list = obstacle_scanner.get_full_obstacle_list(current_time=(kt*self.ts), factor=1.0)
            for i, dyn_obstacle in enumerate(full_dyn_obstacle_list):
                dyn_constraints[i*params_per_dyn_obs:(i+1)*params_per_dyn_obs] = list(itertools.chain(*dyn_obstacle))

            ### Run solver
            for r_id in list(self.robot_dict):
                mode  = self.robot_dict[r_id]['mode']
                color = self.robot_dict[r_id]['color']
                traj_gen:TrajectoryGenerator = self.robot_dict[r_id]['traj_gen']
                other_robot_states = self.get_other_robot_states(r_id)
                current_ref_traj, traj_gen.idx_ref = traj_gen.get_local_ref_traj(traj_gen.idx_ref, traj_gen.ref_traj, traj_gen.state, action_steps=self.config.action_steps, horizon=self.N_hor)
                actions, pred_states, cost = traj_gen.run_step(stc_constraints, dyn_constraints, other_robot_states, current_ref_traj=current_ref_traj, mode=mode) # NOTE: SOLVING HERE
                self.robot_dict[r_id]['pred_states'] = np.array(pred_states)
                ### Plot in loop
                if plot_in_loop:
                    self.plotter.update_plot(r_id, kt, actions[-1], traj_gen.state, cost, np.array(pred_states), current_ref_traj, color=color)
            self.plotter.plot_in_loop(full_dyn_obstacle_list) # plot the dynamic obstacle only once

            ### Prepare for next loop ###
            cnt_done = 0
            for r_id in list(self.robot_dict):
                traj_gen:TrajectoryGenerator = self.robot_dict[r_id]['traj_gen']
                terminated = traj_gen.check_termination_condition(traj_gen.state, traj_gen.past_actions[-1], traj_gen.final_goal)
                if terminated:
                    self.robot_dict[r_id]['done'] = True
                    cnt_done += 1
            if cnt_done == len(self.robot_dict):
                all_terminated = True

            kt += self.config.action_steps

        if self.use_tcp:
            for r_id in list(self.robot_dict):
                traj_gen:TrajectoryGenerator = self.robot_dict[r_id]['traj_gen']
                traj_gen.mng.kill()

        if plot_in_loop:
            self.plotter.show()
            input('Press anything to finish!')
            self.plotter.close()

        return self.robot_dict



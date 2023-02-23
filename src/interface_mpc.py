import itertools

import numpy as np

from pkg_motion_model import motion_model
from mpc_traj_tracker.trajectory_generator import TrajectoryGenerator
from mpc_traj_tracker._path import PathNodeList

from util import utils_geo

from typing import Callable, List, Tuple


DEFAULT_MOTION_MODEL = motion_model.unicycle_model

class InterfaceMpc:
    def __init__(self, config, use_tcp=False, verbose=False, motion_model:Callable=None):
        self._traj_gen = TrajectoryGenerator(config, use_tcp, verbose=verbose)
        motion_model = motion_model if motion_model is not None else DEFAULT_MOTION_MODEL
        self._traj_gen.load_robot_dynamics(motion_model=motion_model)

        self._last_action = np.array([0.0, 0.0])
        self.stc_constraints = [0.0] * self.config.Nstcobs * self.config.nstcobs
        self.dyn_constraints = [0.0] * self.config.Ndynobs * self.config.ndynobs * self.config.N_hor
        self.other_robot_states = [0] * self.config.ns * self.config.N_hor * self.config.Nother

    @property
    def config(self):
        return self._traj_gen.config

    @property
    def state(self):
        return self._traj_gen.state
    
    @property
    def last_action(self):
        return self._last_action
    
    @property
    def goal(self):
        return self._traj_gen.final_goal
    
    @property
    def ref_path(self):
        return self._ref_path

    @property
    def ref_traj(self):
        return self._traj_gen.ref_traj
    
    def set_current_state(self, state: np.ndarray):
        self._traj_gen.set_current_state(state)

    def initialization(self, init_state: np.ndarray, goal_state: np.ndarray, ref_path_list: List[tuple], mode:str='work'):
        self._ref_path = PathNodeList.from_tuples(ref_path_list)
        self._traj_gen.load_init_state(init_state, goal_state)
        self._traj_gen.set_work_mode(mode)
        self._traj_gen.set_ref_trajectory(self.ref_path)
    
    def update_static_constraints(self, obstacle_list):
        for i, map_obstacle in enumerate(obstacle_list):
            b, a0, a1 = utils_geo.polygon_halfspace_representation(np.array(map_obstacle))
            self.stc_constraints[i*self.config.nstcobs : (i+1)*self.config.nstcobs] = (b+a0+a1)

    def update_dynamic_constraints(self, full_dyn_obstacle_list):
        params_per_dyn_obs =self.config.ndynobs * self.config.N_hor
        for i, dyn_obstacle in enumerate(full_dyn_obstacle_list):
            self.dyn_constraints[i*params_per_dyn_obs:(i+1)*params_per_dyn_obs] = list(itertools.chain(*dyn_obstacle))
    
    def update_other_robot_states(self, other_robot_states):
        self.other_robot_states = other_robot_states

    def get_local_ref_traj(self, local_ref_traj:np.ndarray=None) -> Tuple[np.ndarray, np.ndarray]:
        idx_ref = self._traj_gen.idx_ref
        original_ref_traj, idx_ref = self._traj_gen.get_local_ref_traj(idx_ref, self.ref_traj, self.state, action_steps=self.config.action_steps, horizon=self.config.N_hor)
        self._traj_gen.idx_ref = idx_ref
        if local_ref_traj is not None:
            if local_ref_traj.shape[1] == 2:
                local_ref_traj = np.concatenate((local_ref_traj, original_ref_traj[:,[2]]), axis=1)
        return original_ref_traj, local_ref_traj
    
    def get_action(self, current_ref_traj: np.ndarray, mode='work', initial_guess:np.ndarray=None):
        if self._traj_gen.check_termination_condition(self.state, self._last_action, self.goal):
            return None
        actions, pred_states, cost = self._traj_gen.run_step(self.stc_constraints, self.dyn_constraints, 
                                                             self.other_robot_states, current_ref_traj, mode, initial_guess)
        self._last_action = actions[0]
        return actions[0], pred_states, cost
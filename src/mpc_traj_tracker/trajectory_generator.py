# System import
import os
import sys
import math

# External import
import numpy as np

# Custom import 
from util.mpc_config import Configurator
from ._path import PathNode, PathNodeList, TrajectoryNode, TrajectoryNodeList

from typing import Callable, Tuple, Union, List, Dict, Any

'''
File description:
    Generate the trajectory by generating/using the defined MPC solver. 
File content:
    TrajectoryGenerator <class> - Run the MPC problem. Calculate the trajectory step by step.
Comments:
                                                                                      V [MPC] V
    [GPP] --global path & static obstacles--> [LPP] --refernece path & tube width--> [TG(Config)] <--dynamic obstacles-- [OS]
'''

class Solver(): # this is not found in the .so file (in ternimal: nm -D  navi_test.so)
    import opengen as og
    def run(self, p:list, initial_guess, initial_lagrange_multipliers, initial_penalty) -> og.opengen.tcp.solver_status.SolverStatus: pass


class TrajectoryGenerator:
    '''
    Description:
        Generate a smooth trajectory based on the reference path and obstacle information.
        Use a configuration specified by 'utils/config'
    Arguments:
        config  <dotdict> - A dictionary in the dot form contains all information/parameters needed.
        verbose <bool>    - If true, show verbose.
    Attributes:
        __prtname     <str>     - The name to print while running this class.
        config        <dotdict> - As above mentioned.
    Functions
        run <run>  - Run.
    Comments:
        Have fun but may need to modify the dynamic obstacle part (search NOTE).
    '''
    def __init__(self, config:Configurator, use_tcp:bool=False, verbose=False):
        self.__prtname = '[Traj]'
        self.vb = verbose
        self.config = config

        # Common used parameters from config
        self.ts = self.config.ts
        self.ns = self.config.ns
        self.nu = self.config.nu
        self.N_hor = self.config.N_hor

        # Initialization
        self.set_work_mode(mode='safe')
        self.set_obstacle_weights(stc_weights=1e3, dyn_weights=1e3) # default obstacle weights

        self.__import_solver(use_tcp=use_tcp)

    def __import_solver(self, root_dir:str='', use_tcp:bool=False):
        self.use_tcp = use_tcp
        solver_path = os.path.join(root_dir, self.config.build_directory, self.config.optimizer_name)

        import opengen as og
        if not use_tcp:
            sys.path.append(solver_path)
            built_solver = __import__(self.config.optimizer_name) # it loads a .so (shared library object)
            self.solver:Solver = built_solver.solver() # Return a Solver object with run method, cannot find it though
        else: # use TCP manager to access solver
            self.mng:og.opengen.tcp.OptimizerTcpManager = og.tcp.OptimizerTcpManager(solver_path)
            self.mng.start()
            self.mng.ping() # ensure RUST solver is up and runnings

    def load_robot_dynamics(self, motion_model:Callable) -> None:
        """
        motion_model: s'=f(s,a,ts), function that takes in a state and action and returns the next state
        """
        self.motion_model = motion_model

    def load_init_state(self, current_state: np.ndarray, goal_state: np.ndarray):
        if (not isinstance(current_state, np.ndarray)) or (not isinstance(goal_state, np.ndarray)):
            raise TypeError(f'State and action should be numpy.ndarry, got {type(current_state)}/{type(goal_state)}.')
        self.state = current_state
        self.final_goal = goal_state # used to check terminal condition

        self.past_states  = []
        self.past_actions = []
        self.cost_timelist = []
        self.solver_time_timelist = []

        self.idx_ref = 0 # for reference trajectory following

    def set_obstacle_weights(self, stc_weights:Union[list, int], dyn_weights:Union[list, int]):
        '''
        Attribute
            stc_weights [list]: penalty weights for static obstacles (only useful if soft constraints activated)
            dyn_weights [list]: penalty weights for dynamic obstacles (only useful if soft constraints activated)
        '''
        if isinstance(stc_weights, list):
            self.stc_weights = stc_weights
        elif isinstance(stc_weights, (float,int)):
            self.stc_weights = [stc_weights]*self.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(stc_weights)}.')
        if isinstance(dyn_weights, list):
            self.dyn_weights = dyn_weights
        elif isinstance(dyn_weights, (float,int)):
            self.dyn_weights = [dyn_weights]*self.N_hor
        else:
            raise TypeError(f'Unsupported datatype for obstacle weights, got {type(dyn_weights)}.')

    def set_work_mode(self, mode:str='safe'):
        '''
        Attribute
            base_speed: The reference speed
            tuning_params: Penalty parameters for MPC
        '''
        nparams = 10

        ### Base/reference speed
        if mode == 'aligning':
            self.base_speed = self.config.lin_vel_max*self.config.medium_speed
            self.tuning_params = [0.0] * nparams
            self.tuning_params[2] = 100
        else:
            self.tuning_params = [self.config.qpos, self.config.qvel, self.config.qtheta, self.config.lin_vel_penalty, self.config.ang_vel_penalty,
                                  self.config.qpN, self.config.qthetaN, self.config.qrpd, self.config.lin_acc_penalty, self.config.ang_acc_penalty]
            if mode == 'safe':
                self.base_speed = self.config.lin_vel_max*self.config.low_speed
            elif mode == 'work':
                self.base_speed = self.config.lin_vel_max*self.config.high_speed
            elif mode == 'super':
                self.base_speed = self.config.lin_vel_max*self.config.full_speed
            else:
                raise ModuleNotFoundError(f'There is no mode called {mode}.')

    def set_current_state(self, current_state: np.ndarray):
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state

    def set_ref_trajectory(self, ref_path:PathNodeList):
        '''
        Attributes:
            :idx_ref, start from 0
            :ref_traj, global reference trajectory
        '''
        if not isinstance(ref_path, PathNodeList):
            ref_path = PathNodeList([PathNode(x[0], x[1]) for x in ref_path])
        self.idx_ref = 0
        self.ref_traj = self.get_global_ref_traj(self.ts, ref_path, self.state, self.base_speed)

    def check_termination_condition(self, state: np.ndarray, action: np.ndarray, final_goal: np.ndarray) -> bool:
        if np.allclose(state[:2], final_goal[:2], atol=0.05, rtol=0) and abs(action[0]) < 0.05:
            terminated = True
            print(f"{self.__prtname} MPC solution found.")
        else:
            terminated = False
        return terminated


    @staticmethod
    def get_global_ref_traj(ts: float, ref_path:PathNodeList, state: tuple, speed:float) -> TrajectoryNodeList:
        '''
        Description:
            Generate the reference trajectory from the reference path.
        Return:
            ref_traj: x, y coordinates and the heading angles
        '''
        x, y = state[0], state[1]
        x_next, y_next = ref_path[0].x, ref_path[0].y
        
        ref_traj = TrajectoryNodeList([])
        path_idx = 0
        traveling = True
        while(traveling):# for n in range(N):
            while(True):
                dist_to_next = math.hypot(x_next-x, y_next-y)
                if dist_to_next < 1e-9:
                    path_idx += 1
                    x_next, y_next = ref_path[path_idx].x, ref_path[path_idx].y
                    break
                x_dir = (x_next-x) / dist_to_next
                y_dir = (y_next-y) / dist_to_next
                eta = dist_to_next/speed # estimated time of arrival
                if eta > ts: # move to the target node for t
                    x = x+x_dir*speed*ts
                    y = y+y_dir*speed*ts
                    break # to append the position
                else: # move to the target node then set a new target
                    x = x+x_dir*speed*eta
                    y = y+y_dir*speed*eta
                    path_idx += 1
                    if path_idx > len(ref_path)-1 :
                        traveling = False
                        break
                    else:
                        x_next, y_next = ref_path[path_idx][0], ref_path[path_idx][1]
            if not dist_to_next < 1e-9:
                ref_traj.append(TrajectoryNode(x, y, math.atan2(y_dir,x_dir)))
        return ref_traj
    
    @staticmethod
    def get_local_ref_traj(idx_ref:int, ref_traj_global:TrajectoryNodeList, state: tuple, action_steps=1, horizon=20) -> Tuple[np.ndarray, int]:
        '''
        Return:
            :ref_traj_local, every row is a state, the number of rows should be equal to the horizon
        '''
        x_ref     = ref_traj_global.numpy()[:,0].tolist()
        y_ref     = ref_traj_global.numpy()[:,1].tolist()
        theta_ref = ref_traj_global.numpy()[:,2].tolist()

        lb_idx = max(0, idx_ref-1*action_steps)                # reduce search space for closest reference point
        ub_idx = min(len(ref_traj_global), idx_ref+5*action_steps)    # reduce search space for closest reference point

        distances = [math.hypot(state[0]-x[0], state[1]-x[1]) for x in ref_traj_global[lb_idx:ub_idx]]
        idx_next = distances.index(min(distances))

        idx_next += lb_idx  # idx in orignal reference trajectory list
        if (idx_next+horizon >= len(x_ref)):
            tmpx = x_ref[idx_next:]      + [x_ref[-1]]*(horizon-(len(x_ref)-idx_next))
            tmpy = y_ref[idx_next:]      + [y_ref[-1]]*(horizon-(len(y_ref)-idx_next))
            tmpt = theta_ref[idx_next:]  + [theta_ref[-1]]*(horizon-(len(theta_ref)-idx_next))
        else:
            tmpx = x_ref[idx_next:idx_next+horizon]
            tmpy = y_ref[idx_next:idx_next+horizon]
            tmpt = theta_ref[idx_next:idx_next+horizon]
        ref_traj_local = np.array([tmpx, tmpy, tmpt]).transpose()
        return ref_traj_local, idx_next


    def run_step(self, stc_constraints:list, dyn_constraints:list, other_robot_states:list, current_ref_traj:np.ndarray, mode:str='safe', initial_guess:np.ndarray=None):
        '''
        Description:
            Run the trajectory planner for one step.
        Arguments:
            other_robot_states: A list with length "ns*N_hor*Nother" (E.x. [0,0,0] * (self.N_hor*self.config.Nother))
            current_ref_traj: from get_local_ref_traj
        Return:
            :actions
            :pred_states
            :cost
        '''
        ### Mode selection
        # if mode == 'aligning':
        #     if abs(theta_ref[idx_next]-self.state[3])<(math.pi/6):
        #         mode =='work'
        self.set_work_mode(mode)

        ### Get reference states ###
        finish_state = current_ref_traj[-1,:]
        current_refs = current_ref_traj.reshape(-1).tolist()

        ### Get reference velocities ###
        dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
        if dist_to_goal >= self.base_speed*self.N_hor*self.ts:
            speed_ref_list = [self.base_speed]*self.N_hor
        else:
            speed_ref = dist_to_goal / self.N_hor / self.ts
            speed_ref = max(speed_ref, self.config.low_speed)
            speed_ref_list = [speed_ref]*self.N_hor

        if len(self.past_actions):
            last_u = self.past_actions[-1]
        else:
            last_u = np.zeros(self.nu)
            
        ### Assemble parameters for solver & Run MPC###
        params = list(self.state) + list(finish_state) + list(last_u) + \
                 self.tuning_params + current_refs + speed_ref_list + \
                 other_robot_states + \
                 stc_constraints + dyn_constraints + self.stc_weights + self.dyn_weights

        try:
            taken_states, pred_states, actions, cost, solver_time, exit_status = self.run_solver(params, self.state, self.config.action_steps, initial_guess)
        except RuntimeError as err:
            if self.use_tcp:
                self.mng.kill()
            raise RuntimeError(f"Fatal: Cannot run solver. {err}.")

        self.past_states.append(self.state)
        self.past_states += taken_states[:-1]
        self.past_actions += actions
        self.state = taken_states[-1]
        self.cost_timelist.append(cost)
        self.solver_time_timelist.append(solver_time)

        if exit_status in self.config.bad_exit_codes and self.vb:
            print(f"{self.__prtname} Bad converge status: {exit_status}")

        return actions, pred_states, cost

    def run_solver(self, parameters:list, state: np.ndarray, take_steps:int=1, initial_guess:np.ndarray=None):
        '''
        Description:
            Run the solver for the pre-defined MPC problem.
        Argument:
            parameters   <list>:   - All parameters used by MPC, defined in 'build'.
            state        <cs>  :   - The overall states.
            take_steps   <int> :   - The number of control step taken by the input (default 1).
        Return:
            taken_states <list> :  - List of taken states, length equal to take_steps.
            pred_states  <list> :  - List of predicted states at this step, length equal to horizon N.
            actions      <list> :  - List of taken actions, length equal to take_steps.
            cost         <float>:  - The cost value of this step
            solver_time  <float>:  - Time cost for solving MPC of the current time step
            exit_status  <str>  :  - The exit state of the solver.
        Comment:
            The motion model (dynamics) is defined initially.
        '''
        if self.use_tcp:
            return self.run_solver_tcp(parameters, state, take_steps)

        import opengen as og
        solution:og.opengen.tcp.solver_status.SolverStatus = self.solver.run(parameters, initial_guess)
        
        u = solution.solution
        cost:float = solution.cost
        exit_status: str = solution.exit_status
        solver_time: float = solution.solve_time_ms
        
        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model( state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts )
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model( pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts )
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = u[:self.nu*take_steps]
        actions = np.array(actions).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status

    def run_solver_tcp(self, parameters:list, state: np.ndarray, take_steps:int=1):
        solution = self.mng.call(parameters)
        if solution.is_ok():
            # Solver returned a solution
            solution_data = solution.get()
            u = solution_data.solution
            cost: float = solution_data.cost
            exit_status: str = solution_data.exit_status
            solver_time: float = solution_data.solve_time_ms
        else:
            # Invocation failed - an error report is returned
            solver_error = solution.get()
            error_code = solver_error.code
            error_msg = solver_error.message
            self.mng.kill() # kill so rust code wont keep running if python crashes
            raise RuntimeError(f"MPC Solver error: [{error_code}]{error_msg}")

        taken_states:List[np.ndarray] = []
        for i in range(take_steps):
            state_next = self.motion_model( state, np.array(u[(i*self.nu):((i+1)*self.nu)]), self.ts )
            taken_states.append(state_next)

        pred_states:List[np.ndarray] = [taken_states[-1]]
        for i in range(len(u)//self.nu):
            pred_state_next = self.motion_model( pred_states[-1], np.array(u[(i*self.nu):(2+i*self.nu)]), self.ts )
            pred_states.append(pred_state_next)
        pred_states = pred_states[1:]

        actions = u[:self.nu*take_steps]
        actions = np.array(actions).reshape(take_steps, self.nu).tolist()
        actions = [np.array(action) for action in actions]
        return taken_states, pred_states, actions, cost, solver_time, exit_status

    def _vis_params(self, params, current_ref_traj):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        state = params[:self.ns]
        finish_state = params[self.ns:2*self.ns]
        
        ax.plot(state[0], state[1], 'ro', label='start')
        ax.plot(finish_state[0], finish_state[1], 'go', label='finish')
        ax.plot(current_ref_traj[:,0], current_ref_traj[:,1], 'b', label='ref')

        ax.axis('equal')
        ax.legend()
        plt.show()
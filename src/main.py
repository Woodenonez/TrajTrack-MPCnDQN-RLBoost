### System import
import os
import pathlib
import random
import copy

import numpy as np
import matplotlib.pyplot as plt

### DRL import
import gym
from torch import no_grad
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import env_checker

from pkg_dqn.environment import MapDescription, MobileRobot
from pkg_dqn.environment.environment import TrajectoryPlannerEnvironment

### MPC import
from interface_mpc import InterfaceMpc
from util.mpc_config import Configurator

### Helper
from main_pre import generate_map, get_geometric_map, HintSwitcher

### Others
from timer import PieceTimer, LoopTimer

MAX_RUN_STEP = 200


def ref_traj_filter(original: np.ndarray, new: np.ndarray, decay=1):
    filtered = original.copy()
    for i in range(filtered.shape[0]):
        filtered[i, :] = (1-decay) * filtered[i, :] + decay * new[i, :]
        decay *= decay
        if decay < 1e-2:
            decay = 0.0
    return filtered

def load_rl_model_env(generate_map, index: int):
    variant = [
        {
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': False,
            'device': 'auto',
        },
        {
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': False,
            'device': 'cpu',
        },
    ][index]

    if index == 0:
        model_folder_name = 'image'
    elif index == 1:
        model_folder_name = 'ray'
    else:
        raise ValueError('Invalid index')
    model_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'Model', model_folder_name, 'best_model')
    
    env_eval:TrajectoryPlannerEnvironment = gym.make(variant['env_name'], generate_map=generate_map)
    env_checker.check_env(env_eval)
    model = DQN.load(model_path)
    return model, env_eval

def load_mpc(config_path: str):
    config = Configurator(config_path)
    traj_gen = InterfaceMpc(config, motion_model=None) # default motion model is used
    return traj_gen


def main(rl_index:int=1, decision_mode:int=1, to_plot=False):
    """
    Args:
        rl_index: 0 for image, 1 for ray
        decision_mode: 0 for pure rl, 1 for pure mpc, 2 for hybrid
    """
    prt_decision_mode = {0: 'pure_rl', 1: 'pure_mpc', 2: 'hybrid'}
    print(f"The decision mode is: {prt_decision_mode[decision_mode]}")

    time_list = []

    model, env_eval = load_rl_model_env(generate_map, rl_index)

    CONFIG_FN = 'mpc_test.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    traj_gen = load_mpc(cfg_fpath)
    geo_map = get_geometric_map(env_eval.get_map_description(), inflate_margin=0.7)
    traj_gen.update_static_constraints(geo_map.processed_obstacle_list) # assuming static obstacles not changed

    done = False
    with no_grad():
        while not done:
            obsv = env_eval.reset()

            init_state = np.array([*env_eval.agent.position, env_eval.agent.angle])
            goal_state = np.array([*env_eval.goal.position, 0])
            ref_path = list(env_eval.path.coords)
            traj_gen.initialization(init_state, goal_state, ref_path)

            last_mpc_time = 0.0
            last_rl_time = 0.0

            chosen_ref_traj = None
            pred_positions = None            

            for i in range(0, MAX_RUN_STEP):

                if decision_mode == 0:
                    timer_rl = PieceTimer()
                    action_index, _states = model.predict(obsv, deterministic=True)
                    last_rl_time = timer_rl(4, ms=True)
                    obsv, reward, done, info = env_eval.step(action_index)
                    ### Manual step
                    # env_eval.step_obstacles()
                    # env_eval.update_status(reset=False)
                    # obsv = env_eval.get_observation()

                elif decision_mode == 1:
                    env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                             traj_gen.last_action[0], traj_gen.last_action[1])
                    obsv, reward, done, info = env_eval.step(0) # just for plotting and updating status

                    orignal_ref_traj, _ = traj_gen.get_local_ref_traj()
                    chosen_ref_traj = orignal_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj)
                    except Exception as e:
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)
                    if mpc_output is None:
                        break
                    action, pred_states, cost = mpc_output

                elif decision_mode == 2:
                    env_eval.set_agent_state(traj_gen.state[:2], traj_gen.state[2], 
                                             traj_gen.last_action[0], traj_gen.last_action[1])
                    timer_rl = PieceTimer()
                    action_index, _states = model.predict(obsv, deterministic=True)
                    obsv, reward, done, info = env_eval.step(action_index)

                    real_robot:MobileRobot = env_eval.agent
                    pred_positions = []
                    robot_sim = copy.deepcopy(real_robot)
                    for j in range(20):
                        if j == 0:
                            robot_sim.step(action_index, 0.1)
                        else:
                            # robot_sim.step_with_decay_angular_velocity(0.1)
                            robot_sim.step_with_max_speed(0.1)
                        pred_positions.append(list(robot_sim.position))
                    last_rl_time = timer_rl(4, ms=True)
                    
                    orignal_ref_traj, rl_ref_traj = traj_gen.get_local_ref_traj(np.array(pred_positions))
                    filtered_ref_traj = ref_traj_filter(orignal_ref_traj, rl_ref_traj, decay=1) # decay=1 means no decay
                    chosen_ref_traj = filtered_ref_traj
                    timer_mpc = PieceTimer()
                    try:
                        mpc_output = traj_gen.get_action(chosen_ref_traj) # MPC computes the action
                    except Exception as e:
                        print(f'MPC fails: {e}')
                        break
                    last_mpc_time = timer_mpc(4, ms=True)

                else:
                    raise ValueError("Invalid decision mode")
                
                if decision_mode == 0:
                    time_list.append(last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (DQN): {last_rl_time}ms")
                elif decision_mode == 1:
                    time_list.append(last_mpc_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (MPC): {last_mpc_time}ms")
                elif decision_mode == 2:
                    time_list.append(last_mpc_time+last_rl_time)
                    if to_plot:
                        print(f"Step {i}.Runtime (HYB): {last_mpc_time+last_rl_time} = {last_mpc_time}+{last_rl_time}ms")


                if to_plot & (i%10==0): # render every third frame
                    env_eval.render(pred_positions=pred_positions, ref_traj=chosen_ref_traj)

                if i == MAX_RUN_STEP - 1:
                    done = True
                    print('Time out!')
                if done:
                    if to_plot:
                        input('Collision or finish! Press enter to continue...')
                    break

    print(f"Average time ({prt_decision_mode[decision_mode]}): {np.mean(time_list)}ms\n")
    return time_list

if __name__ == '__main__':
    time_list_dqn = main(rl_index=1, decision_mode=0,  to_plot=True)
    time_list_mpc = main(rl_index=1, decision_mode=1,  to_plot=True)
    time_list_hyb = main(rl_index=1, decision_mode=2,  to_plot=True)

    fig, axes = plt.subplots(1,2)

    axes[0].hist(time_list_dqn, bins=10, color='r', alpha=0.5, label='DQN')
    axes[0].hist(time_list_mpc, bins=10, color='b', alpha=0.5, label='MPC')
    axes[0].hist(time_list_hyb, bins=10, color='g', alpha=0.5, label='HYB')
    axes[0].legend()

    axes[1].plot(time_list_dqn, color='r', ls='-', marker='x', label='DQN')
    axes[1].plot(time_list_mpc, color='b', ls='-', marker='x', label='MPC')
    axes[1].plot(time_list_hyb, color='g', ls='-', marker='x', label='HYB')

    plt.show()
    input('Press enter to exit...')
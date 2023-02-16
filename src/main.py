### System import
import os
import pathlib
import random

### DRL import
import gym
from torch import no_grad
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import env_checker

### MPC import
from interface_mpc import InterfaceMpc
from util.mpc_config import Configurator

### Others
from timer import PieceTimer, LoopTimer




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
    
    env_eval = gym.make(variant['env_name'], generate_map=generate_map)
    env_checker.check_env(env_eval)
    model = DQN.load(model_path)
    return model, env_eval

def load_mpc_model_env(config_path: str):
    config = Configurator(config_path)
    traj_gen = InterfaceMpc(config, motion_model=None) # default motion model is used
    return traj_gen


def main(index:int=1):
    CONFIG_FN = 'mpc_default.yaml'
    cfg_fpath = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
    traj_gen = load_mpc_model_env(cfg_fpath)

    model, env_eval = load_rl_model_env(generate_map, index)

    done = False
    with no_grad():
        while not done:
            obsv = env_eval.reset()
            # init_state = list([env_eval.agent.position, env_eval.agent.angle, env_eval.agent.speed, env_eval.agent.angular_velocity])
            for i in range(0, 1000):
                # env_eval.set_agent_state(*init_state)
                # obsv = env_eval.get_observation()
                action, _states = model.predict(obsv, deterministic=True)
                obsv, reward, done, info = env_eval.step(action)

                # real_robot:MobileRobot = env_eval.agent
                # pred_positions = []
                # robot_sim = copy.deepcopy(real_robot)
                # for j in range(20):
                #     if j == 0:
                #         robot_sim.step(action, 0.1)
                #     else:
                #         robot_sim.step_with_decay_angular_velocity(0.1, j+1)
                #     pred_positions.append(list(robot_sim.position))

                if i % 3 == 0: # Only render every third frame for performance (matplotlib is slow)
                    env_eval.render(pred_positions=None)
                if done:
                    break

if __name__ == '__main__':
    main(index=1)
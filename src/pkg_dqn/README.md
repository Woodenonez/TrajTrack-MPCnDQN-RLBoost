# Collision Free Trajectory Planning Using Deep Reinforcement Learning

Please read this entire file before attempting to understand or run the code!

## Getting Started

To install and activate the python environment, install [conda](https://docs.conda.io/en/latest/index.html) and then run

```console
conda env create -f env.yml
conda activate drl-traj-plan
```

[`main.py`](./main.py) contains example code showing how different network models are trained and evaluated.

## Code overview

In order to vary the observation space and reward function, multiple slightly different OpenAI gym environments are created from a common base class.
Each environment variant selects a subset of multiple available components to use.
You can find these variants in the [`environment/variants`](./environment/variants/) folder.
These components (see [`environment/components`](./environment/components/)) either describe a certain kind of environment observation, or a term in the reward function.
For example, [`ReachGoalReward`](./environment/components/reach_goal_reward.py) will add a reward when the agent reaches the goal of the map it is currently in.
More environment variants can be easially created in a similar manner to how the existing variants are defined.

When instansiating any environment, it is neccesarry to pass the `generate_map` argument to the environment constructor.
This function will be called when the environment resets, and is used to generate the map that the agent is placed inside, including the initial position of the agent, goal, obstacle and boundary positions.
Examples of such `generate_map`-functions are found in [`utils/map.py`](./utils/map.py).

### Prioritized experience replay (PER)

PER was hacked in at the last minute into stable-baselines3, the library used for deep reinforcement learning. Classes of Models supporting PER can be found in [`utils/per_dqn.py`](./utils/per_dqn.py).

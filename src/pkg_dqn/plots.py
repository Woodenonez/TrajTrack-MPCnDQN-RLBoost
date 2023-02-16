"""
This file generates all the plots used for the report. It has some extra
dependencies which I can't be bothered to list, but if you happen to have a
Chalmers StuDAT linux computer at hand, this file is comfirmed run on those.
"""
import gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .environment import plot
from .environment import Goal, ATR, Obstacle, Boundary, MapDescription
from .utils.map import generate_map_dynamic, generate_map_corridor

matplotlib.use('pgf')
matplotlib.rc('font', size=8, family='serif', serif=['Times'])
matplotlib.rc('text', usetex=True)

SINGLE_COL = 3.48
DOUBLE_COL = 6


def plot_evaluations(ax: Axes, file: str, *args, **kwargs) -> None:
    N = 1
    data = np.load(file)
    x = data["timesteps"]
    y =  np.mean(data["results"], 1)
    y = np.pad(y, (N // 2, N - 1 - N // 2), mode='edge')
    y = np.convolve(y, np.ones((N,)) / N, mode='valid')
    ax.plot(x, y, *args, **kwargs)


def generate_map() -> MapDescription:
    atr = ATR((3.5, -0.5), 0, 0.85, 0)
    boundary = Boundary([(-2, -2), (8, -2), (8, 10), (-2, 10)])
    obstacles = [
        Obstacle.create_mpc_static([(3, 3), (7.5, 3), (7.5, 6), (3, 6)]),
        Obstacle.create_mpc_static([(-1.5, 7), (1, 7), (1, 9), (-1.5, 9)])    
    ]
    goal = Goal((6, 8))

    return atr, boundary, obstacles, goal

env = gym.make('TrajectoryPlannerEnvironmentRaysReward1-v0', generate_map=generate_map)
env.reset()
env.atr.position -= env.atr.position
env.step(4)


fig, ax = plt.subplots(1, 2, sharey='row', figsize=(SINGLE_COL, 2.4))

plot.obstacles(ax[0], env.obstacles, label=None)
plot.atr(ax[0], env.atr)
plot.reference_path(ax[0], env.path, label=None)
ax[0].set_aspect('equal')
ax[0].set_xlim([-3, 9])
ax[0].set_ylim([-3, 11])
ax[0].set_xticks([], [])
ax[0].set_yticks([], [])

plot.obstacles(ax[1], env.obstacles, padded=True)
ax[1].plot(env.atr.position[0], env.atr.position[1], '.r', label='ATR')
plot.reference_path(ax[1], env.path)
ax[1].set_aspect('equal')
ax[1].set_xlim([-3, 9])
ax[1].set_xticks([], [])
ax[1].set_yticks([], [])

fig.legend(loc='lower center', ncols=2)
fig.savefig("fig/padded_obstacles.svg", bbox_inches='tight')


fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

plot.obstacles(ax, env.obstacles)
plot.atr(ax, env.atr)
plot.reference_path(ax, env.path)

ax.set_aspect('equal')
ax.set_xlim([-3, 9])
ax.set_ylim([-3, 11])
ax.set_xticks([], [])
ax.set_yticks([], [])

closest = np.asarray(env.path.interpolate(env.path_progress).coords)
corners = np.asarray(env.path.coords)[1:,:]
points = np.vstack((closest, corners))
for i in range(points.shape[0]):
    if i == 0:
        ax.plot(points[i, 0], points[i, 1], '.b', label='Closest point')
    else:
        ax.plot(points[i, 0], points[i, 1], '.g', label='Corners' if i == 1 else None)
    
    plot.line(ax, [env.atr.position, points[i, :]], 'k', linewidth=0.5)

ax.text(1.5, -0.2, '$d^\mathrm{p}$')
ax.text(1.8, 1.3, '$d^\mathrm{c}_1$')
ax.text(0.6, 4, '$d^\mathrm{c}_2$')
ax.text(3.8, 4.2, '$d^\mathrm{c}_{3:N}$')

ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

fig.savefig("fig/path_samples.svg", bbox_inches='tight')


fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

plot.boundary(ax, env.boundary)
plot.boundary(ax, env.boundary, ':k', padded=True, label='Padded boundary')
plot.obstacles(ax, env.obstacles)
plot.obstacles(ax, env.obstacles, ':r', padded=True, label='Padded obstacles')
plot.atr(ax, env.atr)
plot.sectors(ax, env.external_obs_component.segments, atr=env.atr, label='Sectors')
plot.rays(ax, env.external_obs_component.rays, atr=env.atr, label='Rays')

ax.set_aspect('equal')
ax.set_xlim([-3, 9])
ax.set_ylim([-3, 11])
ax.set_xticks([], [])
ax.set_yticks([], [])

ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1))

fig.savefig("fig/sector_rays.svg", bbox_inches='tight')


def generate_map_dyn() -> MapDescription:
    atr, boundary, obstacles, goal = generate_map()

    obstacles.append(Obstacle.create_mpc_dynamic((1, 5), (1, 2), 5, 0.5, 0.2, 1))

    return atr, boundary, obstacles, goal

env = gym.make('TrajectoryPlannerEnvironmentImgsReward1-v0', generate_map=generate_map_dyn)
env.reset()
env.atr.position -= env.atr.position
env.step(4)


fig, ax = plt.subplots(1, 2, figsize=(SINGLE_COL, 2.5))

plot.boundary(ax[0], env.boundary)
plot.boundary(ax[0], env.boundary, ':k,', padded=True, label='Padded boundary')
plot.obstacles(ax[0], env.obstacles)
plot.obstacles(ax[0], env.obstacles, ':r', padded=True, label='Padded obstacles')
plot.atr(ax[0], env.atr)
ax[0].set_aspect('equal')
ax[0].set_xlim([-3, 9])
ax[0].set_ylim([-3, 11])
ax[0].set_xticks([], [])
ax[0].set_yticks([], [])

ax[1].imshow(np.flipud(env.external_obs_component.obs.transpose([1, 2, 0])))
ax[1].set_xticks([], [])
ax[1].set_yticks([], [])

fig.legend(loc='lower center', ncols=2)
fig.savefig("fig/image.svg", bbox_inches='tight')


# fig, ax = plt.subplots(2, 1, sharex='col', figsize=(DOUBLE_COL, 4))

# plot_evaluations(ax[0], 'variant-0/evaluations.npz', label='Images')
# plot_evaluations(ax[1], 'variant-1/evaluations.npz', label='Images')
# plot_evaluations(ax[0], 'variant-2/evaluations.npz', label='Rays \& sectors')
# plot_evaluations(ax[1], 'variant-3/evaluations.npz', label='Rays \& sectors')
# plot_evaluations(ax[1], 'variant-4/evaluations.npz', label='Images + PER')

# ax[0].set_title('Reward $R_1$')
# ax[0].set_ylabel('Mean return')
# ax[0].legend()

# ax[1].set_title('Reward $R_2$')
# ax[1].set_ylabel('Mean return')
# ax[1].set_xlabel('Time step $k$')
# ax[1].legend()

# fig.savefig("fig/training.svg", bbox_inches='tight')


fig, ax = plt.subplots(2, 1, sharey='row', figsize=(SINGLE_COL, 2.4))

env = gym.make('TrajectoryPlannerEnvironmentRaysReward1-v0', generate_map=generate_map_dynamic)
env.reset()

plot.boundary(ax[0], env.boundary)
plot.obstacles(ax[0], env.obstacles)
plot.atr(ax[0], env.atr)
plot.reference_path(ax[0], env.path)
ax[0].set_aspect('equal')
ax[0].set_xticks([], [])
ax[0].set_yticks([], [])
ax[0].set_anchor('W')

env = gym.make('TrajectoryPlannerEnvironmentRaysReward1-v0', generate_map=generate_map_corridor)
env.reset()

plot.boundary(ax[1], env.boundary)
plot.obstacles(ax[1], env.obstacles)
plot.atr(ax[1], env.atr)
plot.reference_path(ax[1], env.path)
ax[1].set_aspect('equal')
ax[1].set_xticks([], [])
ax[1].set_yticks([], [])
ax[1].set_anchor('W')
ax[1].legend(loc='upper left', bbox_to_anchor=(1.04, 1))

fig.savefig("fig/training_envs.svg", bbox_inches='tight')


fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))

plot_evaluations(ax, 'variant-2/evaluations.npz', label='$R_1$ + Rays \& sectors')

ax.set_title('Training progress')
ax.set_xlabel('Time step $k$')
ax.set_ylabel('Mean return')
ax.legend()

fig.savefig("fig/catastrophic_forgetting.svg", bbox_inches='tight')

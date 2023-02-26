import os
import pathlib

import numpy as np

from pkg_motion_model import motion_model
from util.mpc_config import Configurator
from mpc_traj_tracker.mpc.mpc_generator import MpcModule

from pkg_path_plan.local_path_plan import LocalPathPlanner
from pkg_path_plan.global_path_plan import GlobalPathPlanner

from scenario_simulator import Simulator
from visualizer.mpc_plot import MpcPlotAfter


### Customize
# CONFIG_FN = 'mpc_test.yaml'
# CONFIG_FN = 'mpc_default.yaml'
CONFIG_FN = 'mpc_longiter.yaml'

INIT_BUILD = False
PLOT_INLOOP = True
show_animation = False
save_animation = False
case_index = 5 # if None, give the hints


### Load configuration
yaml_fp = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'config', CONFIG_FN)
config = Configurator(yaml_fp)

### Build
if CONFIG_FN != 'mpc_default.yaml':
    if INIT_BUILD:
        MpcModule(config).build(motion_model.unicycle_model)

### Load simulator
sim = Simulator(config, case_index, inflate_margin=(config.vehicle_width+config.vehicle_margin))

### Global path
gpp = GlobalPathPlanner(graph=None) # this is not needed here

### Local path
lpp = LocalPathPlanner(sim.graph)

### Load robot
color_list = ['b', 'r', 'g', 'y', 'c', 'm', 'k']
for robot_id in range(len(sim.start)):
    start, end = sim.start[robot_id], sim.waypoints[robot_id][-1]
    ref_path = lpp.get_ref_path(start, end)
    sim.load_robot(robot_id, ref_path, np.array(start), np.array(end), mode='work', color=color_list[robot_id])

### Start & run MPC
playback_dict = sim.run(sim.graph, sim.scanner, plot_in_loop=PLOT_INLOOP)

### Plot results (press any key to continue in dynamic mode if stuck)
# xx, xy     = np.array(state_list)[:,0],  np.array(state_list)[:,1]
# uv, uomega = np.array(action_list)[:,0], np.array(action_list)[:,1]
# plotter = MpcPlotAfter(config, legend_style='single', double_map=False)
# plotter.plot_results(sim.graph, xx, xy, uv, uomega, cost_list, start, end, animation=show_animation, scanner=sim.scanner, video=save_animation)

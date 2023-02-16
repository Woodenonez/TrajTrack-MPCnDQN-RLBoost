import math

import cv2
import numpy as np

# Vis import
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# Type hint only
from matplotlib.axes import Axes
from util.mpc_config import Configurator

N_HOR = 20

def figure_formatter(window_title:str, num_axes_per_column:list, figure_size:tuple=None):
    '''
    Argument
        num_axes_per_column: The length of the list is the number of columns of the figure. 
            E.x. [1,3] means the figure has two columns and with 1 and 3 axes respectively.
        figure_size: If None, then figure size is adaptive.
    Return
        axis_format: List of axes lists, axes[i][j] means the j-th axis in the i-th column.
    '''
    n_col   = len(num_axes_per_column)
    n_row   = np.lcm.reduce(num_axes_per_column) # least common multiple
    row_res = [int(n_row//x) for x in num_axes_per_column] # greatest common divider

    if figure_size is None:
        fig = plt.figure(constrained_layout=True, num=window_title)
    else:
        fig = plt.figure(figsize=figure_size, num=window_title)
        fig.tight_layout()
    # fig.canvas.set_window_title(window_title)
    gs = GridSpec(n_row, n_col, figure=fig)

    axis_format = []
    for i in range(n_col):
        axis_format.append([])
        for j in range(num_axes_per_column[i]):
            row_start = j    *row_res[i]
            row_end   = (j+1)*row_res[i]
            axis_format[i].append(fig.add_subplot(gs[row_start:row_end, i]))
    return fig, gs, axis_format

class MpcPlotInLoop:
    def __init__(self, config:Configurator) -> None:
        '''
        Attribute
            plot_dict_pre   : A dictionary of all plot objects which need to be manually flushed.
            plot_dict_temp  : A dictionary of all plot objects which only exist for one time step.
            plot_dict_inloop: A dictionary of all plot objects which update (append) every time step.
        TODO
            - Methods to flush part of the plot and to destroy an object in case it is not active.
        '''
        self.ts    = config.ts
        self.width = config.vehicle_width

        self.fig, self.gs, axis_format = figure_formatter('PlotInLoop', [3,1])

        self.vel_ax  :Axes = axis_format[0][0]
        self.omega_ax:Axes = axis_format[0][1]
        self.cost_ax :Axes = axis_format[0][2]
        self.path_ax :Axes = axis_format[1][0]

        [ax.autoscale_view(True, True, True) for ax in [self.vel_ax, self.omega_ax, self.cost_ax]] # doesn't work

        self.remove_later = []     # patches need to be flushed
        self.plot_dict_pre = {}    # flush for every life cycle
        self.plot_dict_temp = {}   # flush for every time step
        self.plot_dict_inloop = {} # update every time step, flush for every life cycle

    def show(self):
        self.fig.show()

    def close(self):
        plt.close(self.fig)

    def plot_in_loop_pre(self, map_manager):
        '''
        Description:
            Prepare all axes.
        Argument
            map_manager: with plot(ax) method.
        '''
        [ax.grid('on') for ax in [self.vel_ax, self.omega_ax, self.cost_ax]]
        [ax.set_xlabel('Time [s]') for ax in [self.vel_ax, self.omega_ax, self.cost_ax]]
        self.vel_ax.set_ylabel('Velocity [m/s]')
        self.omega_ax.set_ylabel('Angular velocity [rad/s]')
        self.cost_ax.set_ylabel('Cost')

        map_manager.plot(self.path_ax)
        self.path_ax.set_xlabel('X [m]', fontsize=15)
        self.path_ax.set_ylabel('Y [m]', fontsize=15)
        self.path_ax.axis('equal')
    
    def add_object_to_pre(self, object_id, ref_traj:np.ndarray, start, end, color):
        '''
        Description:
            This function should be called for (new) each object that needs to be plotted.
        Argument
            ref_traj: every row is a state
            color   : Matplotlib style color
        '''
        if object_id in list(self.plot_dict_pre):
            raise ValueError(f'Object ID {object_id} exists!')
        ref_line,  = self.path_ax.plot(ref_traj[:,0], ref_traj[:,1],   color=color, linestyle='--', label='Ref trajectory')
        start_pt,  = self.path_ax.plot(start[0], start[1], marker='*', color=color, markersize=15, alpha=0.2,  label='Start')
        end_pt,    = self.path_ax.plot(end[0],   end[1],   marker='X', color=color, markersize=15, alpha=0.2,  label='End')
        self.plot_dict_pre[object_id] = [ref_line, start_pt, end_pt]

        vel_line,   = self.vel_ax.plot([], [],   marker='o', color=color)
        omega_line, = self.omega_ax.plot([], [], marker='o', color=color)
        cost_line,  = self.cost_ax.plot([], [],  marker='o', color=color)
        past_line,  = self.path_ax.plot([], [],  marker='.', linestyle='None', color=color)
        self.plot_dict_inloop[object_id] = [vel_line, omega_line, cost_line, past_line]

        ref_line_now,  = self.path_ax.plot([], [], marker='x', linestyle='None', color=color)
        pred_line,     = self.path_ax.plot([], [], marker='+', linestyle='None', color=color)
        self.plot_dict_temp[object_id] = [ref_line_now, pred_line]

    def update_plot(self, object_id, kt, action, state, cost, pred_states:np.ndarray, current_ref_traj:np.ndarray, color):
        '''
        Argument
            action[list]     : velocity and angular velocity
            pred_states      : np.ndarray, each row is a state
            current_ref_traj : np.ndarray, each row is a state
        '''
        if object_id not in list(self.plot_dict_pre):
            raise ValueError(f'Object ID {object_id} does not exist!')

        update_list = [action[0], action[1], cost, state]
        for new_data, line in zip(update_list, self.plot_dict_inloop[object_id]):
            if isinstance(new_data, (int, float)):
                line.set_xdata(np.append(line.get_xdata(),  kt*self.ts))
                line.set_ydata(np.append(line.get_ydata(),  new_data))
            else:
                line.set_xdata(np.append(line.get_xdata(),  new_data[0]))
                line.set_ydata(np.append(line.get_ydata(),  new_data[1]))

        temp_list = [current_ref_traj, pred_states]
        for new_data, line in zip(temp_list, self.plot_dict_temp[object_id]):
            line.set_data(new_data[:, 0], new_data[:, 1])

        veh = patches.Circle((state[0], state[1]), self.width/2, color=color, alpha=0.7, label=f'Robot {object_id}')
        self.path_ax.add_patch(veh)
        self.remove_later.append(veh)

    def plot_in_loop(self, dyn_obstacle_list):
        '''
        Argument
            dyn_obstacle_list: list of obstacle_list, where each one has N_hor predictions
        '''
        for obstacle_list in dyn_obstacle_list: # each "obstacle_list" has N_hor predictions
            current_one = True
            for al, pred in enumerate(obstacle_list):
                x,y,rx,ry,angle,alpha = pred
                if current_one:
                    this_color = 'k'
                else:
                    this_color = 'r'
                if alpha > 0:
                    pos = (x,y)
                    this_ellipse = patches.Ellipse(pos, rx*2, ry*2, angle/(2*math.pi)*360, color=this_color, alpha=max(8-al,1)/20, label='Obstacle')
                    self.path_ax.add_patch(this_ellipse)
                    self.remove_later.append(this_ellipse)
                current_one = False

        ### Autoscale
        for ax in [self.vel_ax, self.omega_ax, self.cost_ax]:
            x_min = min(ax.get_lines()[0].get_xdata())
            x_max = max(ax.get_lines()[0].get_xdata())
            y_min = min(ax.get_lines()[0].get_ydata())
            y_max = max(ax.get_lines()[0].get_ydata())
            for line in ax.get_lines():
                if x_min  > min(line.get_xdata()):
                    x_min = min(line.get_xdata())
                if x_max  < max(line.get_xdata()):
                    x_max = max(line.get_xdata())
                if y_min  > min(line.get_ydata()):
                    y_min = min(line.get_ydata())
                if y_max  < max(line.get_ydata()):
                    y_max = max(line.get_ydata())
            ax.set_xlim([x_min, x_max+1e-3])
            ax.set_ylim([y_min, y_max+1e-3])

        plt.draw()
        plt.pause(0.01)
        while not plt.waitforbuttonpress():
            pass

        for j in range(len(self.remove_later)): # robot and dynamic obstacles (predictions)
            self.remove_later[j].remove()
        self.remove_later = []


class MpcPlotAfter:
    def __init__(self, config:Configurator, legend_style, double_map:bool, color_list=None, legend_list=None) -> None:
        '''
        Argument
            legend_style: 'single'(plot one object) or 'compare'(plot multiple objects);
            double_map  : if true, two identical maps are shown (so that one of them can be enlarged)
        TODO
            Modify as MpcPlotInLoop
        '''
        self.ts    = config.ts
        self.width = config.vehicle_width

        self.l_style = legend_style
        self.doub_map = double_map
        self.c_list = color_list
        self.l_list = legend_list

        if not legend_style in ['single', 'compare']:
            raise ValueError(f'The legend style must be "single" or "compare", got {legend_style}.')
        self.__reload()

    def show(self):
        self.fig.show()

    def close(self):
        plt.close(self.fig)

    def __reload(self, make_video=False):
        if make_video:
            fig = plt.figure(figsize=(16,9))
            fig.tight_layout()
        else:
            self.fig = plt.figure(constrained_layout=True)
        if self.doub_map:
            self.gs = GridSpec(6, 4, figure=self.fig)
        else:
            self.gs = GridSpec(3, 4, figure=self.fig)
        self.fig.canvas.set_window_title('PlotAfter')

    def __plot_action(self, ax:Axes, action:list, color='b'): # velocity or angular velocity
        time = np.linspace(0, self.ts*(len(action)), len(action))
        ax.plot(time, action, '-o', markersize = 4, linewidth=2, color=color)

    def __plot_prepare(self, map_manager, start=None, end=None):
        '''Prepare the plot. \\
        Argument
            map_manager: with .plot_map method to plot the map;
        ''' 
        if self.doub_map:
            self.vel_ax   = self.fig.add_subplot(self.gs[0:2, :2])
            self.omega_ax = self.fig.add_subplot(self.gs[2:4, :2])
            self.cost_ax  = self.fig.add_subplot(self.gs[4:6, :2])
            self.path_ax  = self.fig.add_subplot(self.gs[3:, 2:])
            self.path_ax1 = self.fig.add_subplot(self.gs[:3, 2:]) # this is the "double" map
            map_manager.plot_map(self.path_ax)
            map_manager.plot_map(self.path_ax1)
        else:
            self.vel_ax   = self.fig.add_subplot(self.gs[0, :2])
            self.omega_ax = self.fig.add_subplot(self.gs[1, :2])
            self.cost_ax  = self.fig.add_subplot(self.gs[2, :2])
            self.path_ax  = self.fig.add_subplot(self.gs[:, 2:])
            self.path_ax1 = None # if there is no double map
            map_manager.plot_map(self.path_ax)
        
        self.vel_ax.set_ylabel('Velocity [m/s]', fontsize=15)
        self.omega_ax.set_ylabel('Angular velocity [rad/s]', fontsize=15)
        self.cost_ax.set_xlabel('Time [s]', fontsize=15)
        self.cost_ax.set_ylabel('Cost', fontsize=15)
        self.path_ax.set_xlabel('X [m]', fontsize=15)
        self.path_ax.set_ylabel('Y [m]', fontsize=15)

        if self.l_style == 'single':
            legend_elems = [Line2D([0], [0], color='k', label='Original Boundary'),
                            Line2D([0], [0], color='g', label='Padded Boundary'),
                            Line2D([0], [0], color='r', label='Original Obstacles'),
                            Line2D([0], [0], color='y', label='Padded Obstacles'),
                            Line2D([0], [0], marker='o', color='b', label='Generated Path', alpha=0.5),
                            Line2D([0], [0], marker='*', color='g', label='Start Position', alpha=0.5),
                            Line2D([0], [0], marker='*', color='r', label='End Position'),
                            ]
        else:
            legend_elems = [Line2D([0], [0], marker='*', color='g', label='Start Position', alpha=0.5),
                            Line2D([0], [0], marker='*', color='r', label='End Position'),
                            ]
            for c, l in zip(self.c_list, self.l_list):
                legend_elems.append(Line2D([0], [0], marker='o', color=c, label=l, alpha=0.5),)
        self.path_ax.legend(handles=legend_elems, fontsize=15) #, loc='lower left')
        self.path_ax.axis('equal')

        if start is not None:
            self.path_ax.plot(start[0], start[1], marker='*', color='g', markersize=15)
        if end is not None:
            self.path_ax.plot(end[0], end[1], marker='*', color='r', markersize=15)

    def __update_plot(self, xx:list, xy:list, vel:list, omega:list, cost:list, color):
        '''
        Argument
            color[str or ..]: indicate the color in all plots.
        Comment
            All lists (or np.ndarray) should have the same length.
        '''
        self.__plot_action(self.vel_ax,   vel,   self.ts, color)
        self.__plot_action(self.omega_ax, omega, self.ts, color)
        self.__plot_action(self.cost_ax,  cost,  self.ts, color)
        self.path_ax.plot(xx, xy, c=color, marker='o', alpha=0.5)
        if self.path_ax1 is not None:
            self.path_ax1.plot(xx, xy, c=color, marker='o', alpha=0.5)

    def plot_results(self, map_manager, x_coords, y_coords, vel, omega, cost, start, end, animation=False, scanner=None, video=False):
        if animation & (scanner is not None):
            self.plot_dynamic_results(map_manager, x_coords, y_coords, vel, omega, cost, start, end, scanner, video)
        else:
            self.plot_static_results(map_manager, x_coords, y_coords, vel, omega, cost, start, end)

    def plot_static_results(self, map_manager, xx, xy, vel, omega, cost, start=None, end=None):
        self.__plot_prepare(map_manager, start, end)
        self.__plot_action(self.vel_ax,   vel)
        self.__plot_action(self.omega_ax, omega)
        self.__plot_action(self.cost_ax,  cost)
        self.path_ax.plot(xx, xy, c='b', label='Path', marker='o', alpha=0.5)

    def plot_dynamic_results(self, map_manager, xx, xy, vel, omega, cost, start, end, scanner, make_video=False):
        if make_video:
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            self.__reload(make_video=True)
        
        self.__plot_prepare(map_manager, start, end)

        vel_line,   = self.vel_ax.plot(  [1], '-o', markersize=4, linewidth=2)
        omega_line, = self.omega_ax.plot([1], '-o', markersize=4, linewidth=2)
        cost_line,  = self.cost_ax.plot( [1], '-o', markersize=4, linewidth=2) 

        self.vel_ax.set_xlim(0, self.ts * len(xx))
        self.vel_ax.set_ylim(min(vel) - 0.1, max(vel) + 0.1)
        self.vel_ax.grid('on')

        self.omega_ax.set_xlim(0, self.ts * len(xx))
        self.omega_ax.set_ylim(min(omega) - 0.1, max(omega) + 0.1)
        self.omega_ax.grid('on')

        self.cost_ax.set_xlim(0, self.ts * len(xx))
        self.cost_ax.set_ylim(min(cost) - 0.1, max(cost) + 0.1)

        self.path_ax.arrow(start[0], start[1], math.cos(start[2]), math.sin(start[2]), head_width=0.05, head_length=0.1, fc='k', ec='k')
        path_line, = self.path_ax.plot([1], '-ob', alpha=0.7, markersize=5)

        obs        = [object] * scanner.num_obstacles # NOTE: dynamic obstacles
        obs_padded = [object] * scanner.num_obstacles # NOTE: dynamic obstacles
        start_idx = 0
        for i in range(start_idx, len(xx)):
            time = np.linspace(0, self.ts*i, i)
            omega_line.set_data(time, omega[:i])
            vel_line.set_data(time, vel[:i])
            try:
                cost_line.set_data(time, cost[:i])
            except:
                cost_line.set_data(time, cost)
            path_line.set_data(xx[:i], xy[:i])

            veh = patches.Circle((xx[i], xy[i]), self.width/2, color='b', alpha=0.7, label='Robot')
            self.path_ax.add_artist(veh)

            ### Plot obstacles # NOTE
            for idx in range(scanner.num_obstacles): # NOTE: Maybe different if the obstacle is different
                pos = scanner.get_obstacle_info(idx, i*self.ts, 'pos')
                x_radius, y_radius = scanner.get_obstacle_info(idx, i*self.ts, 'radius')
                angle = scanner.get_obstacle_info(idx, i*self.ts, 'angle')

                obs[idx] = patches.Ellipse(pos, x_radius*2, y_radius*2, angle/(2*math.pi)*360, color='r', label='Obstacle')
                x_rad_pad = x_radius + self.width/2
                y_rad_pad = y_radius + self.width/2
                obs_padded[idx] = patches.Ellipse(pos, x_rad_pad*2, y_rad_pad*2, angle/(2*math.pi)*360, color='y', alpha=0.7, label='Padded obstacle')
                
                self.path_ax.add_artist(obs_padded[idx])
                self.path_ax.add_artist(obs[idx])

            ## Plot predictions # NOTE
            pred = []
            for j, obstacle in enumerate(scanner.get_full_obstacle_list(i*self.ts, N_HOR, ts=self.ts)):
                for al, obsya in enumerate(obstacle):
                    x,y,rx,ry,angle,_ = obsya
                    pos = (x,y)
                    this_ellipse = patches.Ellipse(pos, rx*2, ry*2, angle/(2*math.pi)*360, color='r', alpha=max(8-al,1)/20, label='Obstacle')
                    pred.append(this_ellipse)
                    self.path_ax.add_patch(this_ellipse)

            if make_video:
                canvas = FigureCanvas(self.fig) # put pixel buffer in numpy array
                canvas.draw()
                mat = np.array(canvas.renderer._renderer)
                mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
                if i == start_idx:
                    video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (mat.shape[1],mat.shape[0]))
                video.write(mat)
                print(f'\r Wrote frame {i+1}/{len(xx)}    ', end='')
            else:
                plt.draw()
                plt.pause(0.01)
                while not plt.waitforbuttonpress():
                    pass

            veh.remove()
            for j in range(scanner.num_obstacles): # NOTE: dynamic obstacles
                obs[j].remove()
                obs_padded[j].remove()
            for j in range(len(pred)): # NOTE: dynamic obstacles (predictions)
                pred[j].remove()
        
        if make_video:
            video.release()
            cv2.destroyAllWindows()



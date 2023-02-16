import sys
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.axes import Axes


def return_ftd_map():
    """FTD (Factory Traffic Dataset) graph."""
    boundary_coords = [(0,0), (10.0,0), (10.0,2.5), (6.0,2.5), (6.0,4.5), (10.0,4.5), (10.0,6.5), (6.0,6.5), 
                       (6.0,10.0), (4.0,10.0), (4.0,6.5), (0,6.5), (0,4.5), (4.0,4.5), (4.0,2.5), (0,2.5)]
    obstacle_list = [[(5.3,2.3), (5.3,4.5), (5.7,4.5), (5.7,2.5)]]
    return boundary_coords, obstacle_list

def return_crosswalk_map(with_static_obs=True):
    """One crosswalk over a lane connecting two sidewalks.""" 
    boundary_coords = [(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]
    obstacle_list = [[(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                     [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                     [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                     [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    if with_static_obs:
        obstacle_list.append([(3.0, 3.3), (3.0, 3.7), (4.0, 3.7), (4.0, 3.3)])
    crossing_area = [(9.0, 1.5), (11.0, 1.5), (11.0, 8.5), (9.0, 8.5)]
    return boundary_coords, obstacle_list, crossing_area

def return_crossing_map():
    boundary_coords = [(0, 0), (12, 0), (12, 16), (0, 16)]
    obstacle_list = [[(0, 0), (0, 3), (3, 3), (3, 0)],
                     [(0, 9), (0, 12), (3, 12), (3, 9)],
                     [(9, 9), (9, 12), (12, 12), (12, 9)],
                     [(9, 0), (9, 3), (12, 3), (12, 0)]]
    sidewalk_list = [[(0, 3), (0, 4), (4, 4), (4, 0), (3, 0), (3, 3)],
                     [(0, 8), (0, 9), (3, 9), (3, 12), (4, 12), (4, 8)],
                     [(8, 8), (8, 12), (9, 12), (9, 9), (12, 9), (12, 8)],
                     [(8, 0), (8, 4), (12, 4), (12, 3), (9, 3), (9, 0)]]
    crossing_area = [[(4, 3), (4, 4), (8, 4), (8, 3)],
                     [(3, 4), (3, 8), (4, 8), (4, 4)],
                     [(4, 8), (4, 9), (8, 9), (8, 8)],
                     [(8, 4), (8, 8), (9, 8), (9, 4)]]
    return boundary_coords, obstacle_list, sidewalk_list, crossing_area


def plot_crosswalk_map(ax:Axes, with_static_obs=True):
    boundary_coords, obstacle_list, crossing_area = return_crosswalk_map(with_static_obs)
    plot_boundary = np.array(boundary_coords + [boundary_coords[0]])
    ax.plot(plot_boundary[:,0], plot_boundary[:,1], 'k')
    ax.plot([0, 16], [5, 5], c='orange', linestyle='--')
    ax.fill_between([0, 16], [1.6, 1.6], [8.4, 8.4], color='lightgray')
    crossing = patches.Polygon(crossing_area, hatch='-', fc='white', ec='gray')
    ax.add_patch(crossing)
    for obs in obstacle_list:
        obs_edge = obs + [obs[0]]
        xs, ys = zip(*obs_edge)
        ax.plot(xs,ys,'r--')

        obs = np.array(obs)
        poly = patches.Polygon(obs, color='k')
        ax.add_patch(poly)
    ax.axis('equal')

def plot_crossing_map(ax:Axes):
    boundary_coords, obstacle_list, sidewalk_list, crossing_area = return_crossing_map()
    plot_boundary = np.array(boundary_coords + [boundary_coords[0]])
    # Boundary
    ax.plot(plot_boundary[:,0], plot_boundary[:,1], 'k')
    # Lane
    ax.plot([0, 12], [6, 6], c='orange', linestyle='--')
    ax.plot([6, 6], [0, 12], c='orange', linestyle='--')
    ax.fill_between([0, 12], [4, 4], [8, 8], color='lightgray')
    ax.fill_between([4, 8], [0, 0], [12, 12], color='lightgray')
    # Area
    for cs in crossing_area:
        cs_edge = cs + [cs[0]]
        xs, ys = zip(*cs_edge)
        plt.plot(xs,ys,'gray')
        poly = patches.Polygon(np.array(cs), hatch='-', color='white')
        ax.add_patch(poly)
    for sw in sidewalk_list:
        poly = patches.Polygon(np.array(sw), color='gray')
        ax.add_patch(poly)
    for obs in obstacle_list:
        obs_edge = obs + [obs[0]]
        xs, ys = zip(*obs_edge)
        plt.plot(xs,ys,'k')
        poly = patches.Polygon(np.array(obs), color='k')
        ax.add_patch(poly)
    ax.axis('equal')
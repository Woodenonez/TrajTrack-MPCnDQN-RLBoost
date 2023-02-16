import math
from typing import Tuple, Union

import numpy as np
import networkx as nx
import matplotlib.patches as patches
from matplotlib.axes import Axes

from util.basic_object import *
from util.maptype import GeometricMap, OccupancyMap

'''
All supported scenes:
1. Geometric maps
    - "crosswalk"
    - "crossroads"
2. Occupancy maps
    - "bookstore"
    - "warehouse"
'''

GEOMETRIC_MAP_SCENES = ['crosswalk', 'crossroads']
OCCUPANCY_MAP_SCENES = ['bookstore', 'warehouse', ]

class SceneGraph:
    def __init__(self, scene:str, map_type:str, map_info:dict):
        self.scene = scene
        self.map_type = map_type.lower()
        self.map_info = map_info
        if self.map_type == 'geometric':
            self.base_map = GeometricMap(map_info['boundary'], map_info['obstacle_list'], map_info['inflation'])
        elif self.map_type == 'occupancy':
            self.base_map = OccupancyMap(map_info['map_image'], map_info['threshold'])
        else:
            raise ValueError(f'Map type must be "geometric" or "occupancy", got {map_type}.')

        node_dict, edge_list = return_network_element(scene)
        self.NG = NetGraph(node_dict, edge_list)
        self.NG.set_distance_weight()

    def return_random_path(self, start_node_index, num_traversed_nodes:int):
        return self.NG.return_random_path(start_node_index, num_traversed_nodes)

    def plot_map(self, ax:Axes):
        if self.map_type == 'geometric':
            plot_geometric_map(ax, self.map_info)
        elif self.map_type == 'occupancy':
            ax.imshow(self.base_map(), cmap='Greys')

    def plot_path(self, ax:Axes, path:PathNodeList, style='k--'):
        np_path = np.array(path())
        ax.plot(np_path[:,0], np_path[:,1], style)

    def plot_netgraph(self, ax:Axes, node_style='x', node_text:bool=True, edge_color='r'):
        self.plot_map(ax)
        self.plot_netgraph_nodes(ax, node_style, node_text)
        self.plot_netgraph_edges(ax, edge_color)

    def plot_netgraph_nodes(self, ax:Axes, style='x', with_text=True):
        [ax.plot(self.NG.get_x(n), self.NG.get_y(n), style) for n in list(self.NG.nodes)]
        if with_text:
            [ax.text(self.NG.get_x(n), self.NG.get_y(n), n) for n in list(self.NG.nodes)]

    def plot_netgraph_edges(self, ax:Axes, edge_color='r'):
        nx.draw_networkx_edges(self.NG.G, nx.get_node_attributes(self.NG.G,'pos'), ax=ax, edge_color=edge_color)

#%%## Map Network 
def return_network_element(scene:str) -> Tuple[dict, list]:
    if scene not in GEOMETRIC_MAP_SCENES + OCCUPANCY_MAP_SCENES:
        raise NameError(f'Scene {scene} does not exist.')
    if scene in OCCUPANCY_MAP_SCENES:
        map_type = 'occupancy'
    else:
        map_type = 'geometric'

    if scene=='bookstore':
        node_dict, edge_list = return_bookstore_element()
    elif scene=='warehouse':
        node_dict, edge_list = return_warehouse_element()
    else:
        return None, map_type # some scenes don't have network graph
    return node_dict, edge_list


def return_bookstore_element(rescale:float=3) -> Tuple[dict, list]:
    nodes1 = {26:(20, 45),  1:(20, 285), 2:(20, 450)}
    nodes2 = {3:(70, 450), 4:(130,450), 5:(190,450), 6:(260,450), 7:(320,450), 8:(380,450), 9:(485,450)}
    nodes3 = {10:(70, 355), 11:(130,355), 12:(190,355), 13:(260,355), 14:(320,355), 15:(380,355)}
    nodes4 = {16:(130,260), 17:(190,260), 18:(250,260), 19:(320,260), 20:(380,260)}
    nodes5 = {21:(130, 45), 22:(240, 85), 23:(130,180), 24:(250,180), 25:(410,180)}
    nodes = {**nodes1, **nodes2, **nodes3, **nodes4, **nodes5}
    for key, value in nodes.items():
        nodes[key] = [x*rescale for x in value] # the original scale is 500*500
    edges = [(26,1), (1,2), (1,10), (1,16),
             (2,3), (3,4), (3,10), (4,5), (4,11), (5,6), (5,12), (6,7), (6,13), (7,8), (7,14), (8,9), (8,15),
             (10,11), (11,12), (11,16), (12,18), (12,17), (13,14), (13,17), (13,18), (14,15), (14,19), (15,20),
             (16,17), (17,18), (18,19), (18,24), (19,20), (20,25), (22,21), (24,22), (24,23), (24,25), (21,26), ]
    return nodes, edges

def return_warehouse_element(rescale:float=1) -> Tuple[dict, list]:
    nodes = {1:(110, 20), 2:(110,75), 3:(110,103), 4:(110,138), 5:(110,165), 6:(110,195), 7:(110, 250), 
             8:(160, 20), 9:(160,75), 10:(160,103), 32:(160,120), 11:(160,138), 12:(160,165), 13:(160,210), 14:(160,250), 
             15:(235,20), 16:(235,120), 17:(235,175), 18:(235,210), 19:(235,250), 
             20:(255, 20), 21:(255, 145), 22:(255,175), 23:(255,200), 24:(255,220), 25:(255,250), 
             26:(300,20), 27:(300,115), 28:(310,145), 29:(310,175), 30:(310,200), 31:(310,250), 
             }
    for key, value in nodes.items():
        nodes[key] = [x*rescale for x in value]
    edges = [(1,2), (1,8), (2,3), (2,9), (3,4), (3,10), (4,5), (4,11), (5,6), (5,12), (6,7), (6,13), (7,14),
             (8,9), (8,15), (9,10), (10,32), (32,16), (11,12), (11,32), (12,13), (12,17), (13,14), (13,18), (14,19),
             (15,16), (15,20), (16,17), (16,21), (16,27), (17,18), (17,22), (18,19), (18,23), (18,24), (19,25), 
             (20,21), (20,26), (21,22), (21,28), (22,23), (22,29), (23,24), (23,30), (24,25), (25, 31),
             (26,27), (27,28), (28,29), (29,30), (30,31),
             (23,31), (25,30), (24,30), (24,31)
             ]
    return nodes, edges

def return_warehouse_old_element(rescale:float=0.4) -> Tuple[dict, list]:
    nodes = {40:(137, 12), 1:(137, 80), 2:(137,150), 3:(137,237), 4:(137,305), 5:(137,380), 6:(137,480), 
             7:(180, 12), 8:(180, 80), 9:(180,150), 10:(180,237), 11:(185,305), 12:(185,380), 13:(185,480), 
             14:(240,80), 15:(240,150), 16:(240,237), 17:(240,305), 18:(240,380), 19:(240,480), 
             20:(300, 8), 21:(300, 65), 22:(300,126), 23:(300, 200), 
             24:(305,260), 25:(305,305), 26:(305,350), 27:(305,395), 28:(305,442), 29:(305,487),
             30:(425, 8), 31:(425, 65), 32:(425,126), 33:(425, 200), 
             34:(425,260), 35:(425,305), 36:(425,350), 37:(425,395), 38:(425,442), 39:(425,487)
             }
    for key, value in nodes.items():
        nodes[key] = [x*rescale for x in value]
    edges = [(40,1), (40,7), (1,2), (1,8), (2,3), (2,9), (3,4), (3,10), (4,5), (4,11), (5,6), (5,12), (6,13), 
             (7,8), (8,9), (8,14), (9,10), (9,15), (10,11), (10,16), (11,12), (11,17), (12,13), (12,18), (13,19), 
             (14,15), (14,21), (15,16), (15,22), (16,17), (16,23), (16,24), (17,18), (17,25), (18,19), (18,26), (18,27), (19,28), (19,29),
             (20,21), (20,30), (21,22), (21,31), (22,23), (22,32), (23,24), (23,33), (24,25), (24,34),
             (25,26), (25,35), (26,27), (26,36), (27,28), (27,37), (28,29), (28,38), (29,39), 
             (30,31), (31,32), (32,33), (33,34), (34,35), (35,36), (36,37), (37,38), (38,39),
             ]
    return nodes, edges

#%%## Map info
def return_map_info(scene:str) -> Tuple[dict, str]:
    if scene not in GEOMETRIC_MAP_SCENES + OCCUPANCY_MAP_SCENES:
        raise NameError(f'Scene {scene} does not exist.')
    if scene in OCCUPANCY_MAP_SCENES:
        map_type = 'occupancy'
    else:
        map_type = 'geometric'

    if scene=='crosswalk':
        boundary_coords, obstacle_list, extra_info = return_crosswalk_info()
    elif scene=='crossroads':
        boundary_coords, obstacle_list, extra_info = return_crossroads_info()
    else:
        raise NameError(f'Scene {scene} is under construction.')

    map_info = {'boundary':boundary_coords, 'obstacle_list':obstacle_list}
    if extra_info is not None:
        for key, value in extra_info.items():
            map_info[key] = value

    return map_info, map_type

def return_crosswalk_info() -> Tuple[list, list, Union[None, dict]]:
    boundary_coords = [(0, 0), (16, 0), (16, 10), (0, 10)]  # in counter-clockwise ordering
    obstacle_list = [[(0, 1.5), (0, 1.6), (9, 1.6), (9, 1.5)],
                     [(0, 8.4), (0, 8.5), (9, 8.5), (9, 8.4)],
                     [(11, 1.5), (11, 1.6), (16, 1.6), (16, 1.5)],
                     [(11, 8.4), (11, 8.5), (16, 8.5), (16, 8.4)]] # in clock-wise ordering
    crossing_area = [(9, 1.5), (11, 1.5), (11, 8.5), (9, 8.5)]
    return boundary_coords, obstacle_list, {'crosswalk': crossing_area}

def return_crossroads_info() -> Tuple[list, list, Union[None, dict]]:
    boundary_coords = [(0, 0), (12, 0), (12, 16), (0, 16)]  # in counter-clockwise ordering
    obstacle_list = [[(0, 0), (0, 3), (3, 3), (3, 0)],
                     [(0, 9), (0, 12), (3, 12), (3, 9)],
                     [(9, 9), (9, 12), (12, 12), (12, 9)],
                     [(9, 0), (9, 3), (12, 3), (12, 0)]] # in clock-wise ordering
    sidewalk_list = [[(0, 3), (0, 4), (4, 4), (4, 0), (3, 0), (3, 3)],
                     [(0, 8), (0, 9), (3, 9), (3, 12), (4, 12), (4, 8)],
                     [(8, 8), (8, 12), (9, 12), (9, 9), (12, 9), (12, 8)],
                     [(8, 0), (8, 4), (12, 4), (12, 3), (9, 3), (9, 0)]]
    crossing_area = [[(4, 3), (4, 4), (8, 4), (8, 3)],
                     [(3, 4), (3, 8), (4, 8), (4, 4)],
                     [(4, 8), (4, 9), (8, 9), (8, 8)],
                     [(8, 4), (8, 8), (9, 8), (9, 4)]]
    return boundary_coords, obstacle_list, {'crosswalk': crossing_area, 'sidewalk': sidewalk_list}

# ------------------------------

# def return_bookstore_info() -> :


#%%## Map visualization
def plot_geometric_map(ax:Axes, map_info:dict, start:tuple=None, end:tuple=None):
    boundary_coords = map_info['boundary']
    obstacle_list = map_info['obstacle_list']

    boundary = np.array(boundary_coords + [boundary_coords[0]])
    ax.plot(boundary[:,0], boundary[:,1], 'k--')
    for obs in obstacle_list:
        obs_edge = obs + [obs[0]]
        xs, ys = zip(*obs_edge)
        ax.plot(xs,ys,'r-', linewidth=1)
        poly = patches.Polygon(np.array(obs), color='k')
        ax.add_patch(poly)
    if start is not None:
        ax.plot(start[0], start[1], 'r*')
    if end is not None:
        ax.plot(end[0], end[1], 'g*')
    ax.axis('equal')

def plot_crosswalk_map(ax:Axes, map_info:dict, start:tuple=None, end:tuple=None):
    plot_geometric_map(ax, map_info, start, end)
    ax.plot([0, 16], [5, 5], c='orange', linestyle='--')
    ax.fill_between([0, 16], [1.6, 1.6], [8.4, 8.4], color='lightgray')
    crossing = patches.Polygon(map_info['crosswalk'], hatch='-', fc='white', ec='gray')
    ax.add_patch(crossing)

def plot_crossroads_map(ax:Axes, map_info:dict, start:tuple=None, end:tuple=None):
    plot_geometric_map(ax, map_info, start, end)
    # Lane
    ax.plot([0, 12], [6, 6], c='orange', linestyle='--')
    ax.plot([6, 6], [0, 12], c='orange', linestyle='--')
    ax.fill_between([0, 12], [4, 4], [8, 8], color='lightgray')
    ax.fill_between([4, 8], [0, 0], [12, 12], color='lightgray')
    # Area
    for cs in map_info['crosswalk']:
        cs_edge = cs + [cs[0]]
        xs, ys = zip(*cs_edge)
        ax.plot(xs,ys,'gray')
        poly = patches.Polygon(np.array(cs), hatch='-', color='white')
        ax.add_patch(poly)
    for sw in map_info['sidewalk']:
        # sw_edge = sw + [sw[0]]
        # xs, ys = zip(*sw_edge)
        # plt.plot(xs,ys,'k')
        poly = patches.Polygon(np.array(sw), color='gray')
        ax.add_patch(poly)
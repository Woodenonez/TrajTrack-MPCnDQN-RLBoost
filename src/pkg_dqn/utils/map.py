"""
This file is used to generate the maps which the DRL agent will be trained and
evaluated in. Such a map constitutes e.g. the initial robot position, the goal
position and the locations of obstacles and boundaries.
"""

import math
import random

import numpy as np
import shapely.ops
from shapely.geometry import LineString, Polygon, JOIN_STYLE, Point

from ..environment import MobileRobot, Obstacle, Boundary, Goal, MapDescription, MapGenerator

from typing import Union, List, Tuple

### Training maps ###

def generate_map_mpc(i: Union[int, None] = None) -> MapGenerator:
    """
    Generates maps from the paper https://doi.org/10.1109/CASE49439.2021.9551644
    on MPC.

    ::returns:: When the parameter ``i`` is specified, returns a MapGenerator which
    generates the map with index ``i`` from the MPC paper. When ``i`` is
    ``None``, returns a MapGenerator creating a random map from the MPC paper.
    """
    envs = [
        {
            'boundary': [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            'static_obstacles': [[(3.0, 3.0), (3.0, 7.0), (7.0, 7.0), (7.0, 3.0)]],
            'start': (1, 1, math.radians(0)),
            'goal': (8, 8, math.radians(90)),
        },
        {
            'boundary': [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
            'static_obstacles': [
                [(5.0, 0.0), (5.0, 15.0), (7.0, 15.0), (7.0, 0.0)],
                [(12.0, 12.5), (12.0, 20.0), (15.0, 20.0), (15.0, 12.5)], 
                [(12.0, 0.0), (12.0, 7.5), (15.0, 7.5), (15.0, 0.0)]],
            'start': (1, 5, math.radians(45)),
            'goal': (19, 10, math.radians(0)),
        },
        {
            'boundary': [(0.0, 0.0), (10.0, 0.0), (10.0 ,10.0), (25.0, 10.0), (25.0, 0.0), (50.0, 0), (50, 50), (0, 50), (0, 16), (10, 16), (10, 45), (15, 45), (15, 30), (35, 30), (35, 15), (0, 15)],
            'static_obstacles': [
                [(30.0, 5.0), (30.0, 14.5), (40.0, 14.5), (40, 5.0)],
                [(45.0, 15.0), (44.0, 20.0), (46.0, 20.0)],
                [(25, 35), (25, 40), (40, 40), (40, 35)],
                [(32.0, 6.0), (32.0, 10.5), (42.0, 12.5), (42, 8.0)]],
            'start': (1, 1, math.radians(225)),
            'goal': (5, 20, math.radians(270)),
            'dynamic_obstacles': [
                [[17.5, 43], [22, 37.5], 0.1, 0.2, 0.5, 0.1], 
                [[40.5, 18], [37, 26], 0.1, 0.5, 0.2, 0.5],
                [[6.5, 5], [4.5, 7], 0.1, 0.5, 1, 2]],
        },
        {
            'boundary': [(3.6, 57.8), (3.6, 3.0), (58.3, 3.0), (58.1, 58.3)],
            'static_obstacles': [
                [(21.1, 53.1), (21.4, 15.1), (9.3, 15.1), (9.1, 53.1)],
                [(35.7, 52.2), (48.2, 52.3), (48.7, 13.6), (36.1, 13.8)], 
                [(17.0, 50.5), (30.7, 50.3), (30.6, 45.0), (17.5, 45.1)],
                [(26.4, 39.4), (40.4, 39.3), (40.5, 35.8), (26.3, 36.0)],
                [(19.3, 31.7), (30.3, 31.6), (30.1, 27.7), (18.9, 27.7)],
                [(26.9, 22.7), (41.4, 22.6), (41.1, 17.5), (27.4, 17.6)]],
            'start': (30, 5, math.radians(90)),
            'goal': (30, 55, math.radians(90)),
        },
        {
            'boundary': [(54.0, 57.8), (7.8, 57.5), (7.5, 17.9), (53.0, 17.0)],
            'static_obstacles': [
                [(14.0, 57.6), (42.1, 57.6), (42.2, 52.0), (13.4, 52.0)], 
                [(7.7, 49.1), (32.2, 49.0), (32.1, 45.3), (7.7, 45.8)], 
                [(34.2, 53.0), (41.2, 53.1), (40.9, 31.7), (34.4, 31.9)], 
                [(35.7, 41.7), (35.7, 36.8), (11.7, 39.8), (12.1, 44.0), (31.3, 43.3)], 
                [(5.8, 37.6), (24.1, 35.0), (23.6, 29.8), (5.0, 31.8)], 
                [(27.1, 39.7), (32.7, 39.0), (32.8, 24.7), (16.2, 20.9), (14.5, 25.9), (25.3, 26.7), (27.9, 31.4), (26.1, 39.2)]],
            'start': (10.3, 55.8, math.radians(270)),
            'goal': (38.1, 25.0, math.radians(300)),
        },
        {
            'boundary': [(0.37, 0.32), (5.79, 0.31), (5.79, 5.18), (0.14, 5.26)],
            'static_obstacles': [[(2.04, 0.28), (2.0, 3.8), (2.8, 3.81), (2.78, 0.29)]],
            'start': (1.01, 0.98, math.radians(90)),
            'goal': (3.82, 1.05, math.radians(270)),
        },
        {
            'boundary': [(1.55, 1.15), (29.0, 1.1), (29.0, 28.75), (0.85, 28.9), (0.85, 1.15)],
            'static_obstacles': [[(5.6, 3.3), (5.75, 20.15), (18.35, 20.05), (18.35, 19.7), (7.25, 19.7), (7.05, 3.2)], [(13.85, 23.4), (21.25, 23.35), (21.1, 16.4), (6.9, 16.35), (6.7, 12.9), (23.45, 13.25), (23.4, 25.05), (13.0, 25.35)]],
            'start': (2.95, 13.5, math.radians(90)),
            'goal': (9.6, 18.1, math.radians(180)),
        },
        {
            'boundary': [(2.0, 1.08), (22.8, 1.12), (22.84, 19.16), (1.8, 19.24)],
            'static_obstacles': [[(9.64, 5.28), (9.56, 10.72), (8.68, 11.88), (9.48, 12.2), (10.52, 10.96), (11.6, 12.12), (12.6, 11.36), (11.28, 10.4), (11.6, 0.56), (9.68, 0.68)]],
            'start': (7.16, 8.16, math.radians(90)),
            'goal': (12.72, 9.32, math.radians(265)),
        },
        {
            'boundary': [(0.96, 1.88), (22.88, 1.72), (22.92, 20.8), (0.64, 20.92)],
            'static_obstacles': [[(9.12, 1.48), (8.8, 9.56), (9.76, 12.72), (10.8, 9.56), (11.08, 1.48)]],
            'start': (7.44, 6.16, math.radians(90)),
            'goal': (12.44, 6.4, math.radians(265)),
        },
        {
            'boundary': [(2.36, 1.6), (22.6, 1.84), (22.16, 21.04), (1.52, 20.88)],
            'static_obstacles': [[(9.92, 1.24), (9.64, 8.52), (12.6, 10.44), (15.6, 8.76), (15.76, 1.08)]],
            'start': (7.08, 5.88, math.radians(90)),
            'goal': (17.8, 6.56, math.radians(265)),
        },
        {
            'boundary': [(1.5, 1.0), (1.7, 58.6), (59.0, 58.4), (58.6, 1.3)],
            'static_obstacles': [
                [(27.0, 6.0), (27.0, 33.0), (4.0, 33.0), (4.0, 6.0)], 
                [(65.0, 6.0), (28.1, 6.0), (28.1, 33.0), (65.0, 33.0)], 
                [(4.4, 34.1), (44.0, 34.1), (44.0, 39.3), (55.3, 39.6), (55.3, 42.8), (44.0, 42.3), (44.1, 49.1), (54.9, 49.2), (54.9, 53.0), (4.7, 53.0)], 
                [(47.7, 36.2), (47.7, 34.6), (57.8, 34.5), (57.8, 36.3)]],
            'start': (27.8, 2.7, math.radians(90)),
            'goal': (50.3, 45.9, math.radians(0)),
        },
        {
            'boundary': [(11.9, 3.6), (11.9, 50.6), (47.3, 50.6), (47.3, 3.6)],
            'static_obstacles': [
                [(11.9, 11.8), (22.2, 11.8), (22.2, 15.9), (11.9, 15.9)],
                [(11.9, 20.4), (22.2, 20.4), (22.2, 25.0), (11.9, 25.0)],
                [(28.0, 25.5), (28.0, 3.6), (37.8, 3.6), (37.8, 25.5)], 
                [(15.9, 29.5), (37.7, 29.5), (37.7, 31.7), (15.9, 31.7)],
                [(37.7, 31.7), (37.7, 44.5), (35.0, 44.5), (35.0, 31.7)],
                [(25.3, 44.5), (25.3, 40.7), (35.0, 40.7), (35.0, 44.5)],
                [(29.8, 28.7), (29.8, 25.8), (34.5, 25.8), (34.5, 28.7)]],
            'start': (18.9, 7.0, math.radians(45)),
            'goal': (44.7, 6.8, math.radians(270)),
            'dynamic_obstacles': [
                [[18.5, 18.2],[28.1, 18.2], 0.06, 0.5, 1.0, math.pi / 2],
                [[16.775, 34.0], [22.5, 42.2], 0.07, 0.3, 0.7, math.pi / 2 + 0.961299],
                [[44.3, 9.2], [40.5, 31.8], 0.0745, 0.6, 0.6, 0]],
        }
    ]

    def generate_map() -> MapDescription:
        env = random.choice(envs) if i is None else envs[i]

        boundary = Boundary(env['boundary'])
        obstacles = [
            *[Obstacle.create_mpc_static(coords) for coords in env.get('static_obstacles', [])],
            *[Obstacle.create_mpc_dynamic(*args) for args in env.get('dynamic_obstacles', [])]]
        goal = Goal(env['goal'][:2])
        init_state = np.array(env['start'][:2] + (env['start'][2]+random.uniform(-0.2, 0.2), 0, 0))
        robot = MobileRobot(init_state)

        return robot, boundary, obstacles, goal

    return generate_map


def generate_map_dynamic() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstalces
    """

    init_state = np.array([5, random.uniform(5, 15), random.uniform(0, 2*math.pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (40, 0), (40, 20), (0, 20)])
    obstacles = []
    for i in range(10):
        x = random.uniform(10, 30)
        y = random.uniform(0, 20)

        if i < 3:
            w = max(4, random.uniform(0, 0.5 * min(x - 10, 30 - x)))
            h = max(4, random.uniform(0, min(y, 20 - y)))
            x0 = x - w/2
            y0 = y - h/2

            obstacles.append(Obstacle.create_mpc_static([(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]))
        else:
            x2 = x + random.uniform(-5, 5)
            y2 = y + random.uniform(-5, 5)
            rx = random.uniform(0.2, 1.2)
            ry = random.uniform(0.2, 1.2)
            freq = random.uniform(0.3, 0.7)
            angle = random.uniform(0, 2 * math.pi)

            obstacles.append(Obstacle.create_mpc_dynamic((x, y), (x2, y2), freq, rx, ry, angle))
    goal = Goal((35, random.uniform(5, 15)))

    return robot, boundary, obstacles, goal


def generate_map_corridor() -> MapDescription:
    """
    Generates a map with a randomized narrow corridor ando no dynamic obstacles.
    """

    max_angle = math.pi / 2

    wall_padding = 5
    corridor_padding = random.uniform(0.7, 1.5)

    coords = np.asarray([(0, 0), (wall_padding, 0)])
    angle = 0
    for i in range(3):
        lo = -max_angle - angle
        hi = max_angle - angle
        dangle = random.uniform(lo, hi)
        dangle = dangle**2 / (hi if dangle > 0 else lo)
        angle += dangle

        length = random.uniform(2, 8)
        coords = np.vstack((coords, coords[i + 1, :] + length * np.asarray((math.cos(angle), math.sin(angle)))))
    coords = np.vstack((coords, coords[-1, :] + (wall_padding, 0)))

    init_state = np.array([coords[0, 0], coords[0, 1], random.uniform(0, 2 * math.pi), 0, 0])
    robot = MobileRobot(init_state)
    goal = Goal(coords[-1])

    corridor = LineString(coords)
    minx, miny, maxx, maxy = corridor.bounds

    wall_padding = 5
    pminx = minx - wall_padding
    pminy = miny - wall_padding
    pmaxx = maxx + wall_padding
    pmaxy = maxy + wall_padding

    boundary = Boundary([(pminx, pminy), (pmaxx, pminy), (pmaxx, pmaxy), (pminx, pmaxy)])

    pminx = minx + wall_padding
    pmaxx = maxx - wall_padding

    obstacles = []
    if pminx < pmaxx:
        box = Polygon([(pminx, pminy), (pmaxx, pminy), (pmaxx, pmaxy), (pminx, pmaxy)])
        left = corridor.parallel_offset(corridor_padding, 'left', join_style=JOIN_STYLE.mitre, mitre_limit=1)
        right = corridor.parallel_offset(corridor_padding, 'right', join_style=JOIN_STYLE.mitre, mitre_limit=1)

        eps = 1e-3

        split = shapely.ops.split(box, right)
        test = Point((pminx + eps, pminy + eps))
        for geom in split.geoms:
            if geom.contains(test):
                obstacles.append(Obstacle.create_mpc_static(geom.exterior.coords[:-1]))
                break
        
        split = shapely.ops.split(box, left)
        test = Point((pminx + eps, pmaxy - eps))
        for geom in split.geoms:
            if geom.contains(test):
                obstacles.append(Obstacle.create_mpc_static(geom.exterior.coords[:-1]))
                break
    
    return robot, boundary, obstacles, goal


### Test maps ###
"""
Scene 1: Crosswalk
    - a. Single rectangular static obstacle (small, medium, large)
    - b. Two rectangular static obstacles (small/large stagger, close/far aligned)
    - c. Single non-convex static obstacle (deep/shallow u-/v-shape)
    - d. Single dynamic obstacle (crash, cross)
    --------------------
                |   |
    R =>    S   | D |
    --------------------

Scene 2: Turning
    - a. Single rectangular obstacle (right, sharp, u-shape)
    |-------|
    |   ->  \
    | R |\   \
"""

test_scene_1_dict = {1: [1, 2, 3], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2]}
test_scene_2_dict = {1: [1, 2, 3]}

def generate_map_scene_1(sub_index: int, scene_option: int) -> MapDescription:
    """
    Subscene index (`sub_index`) with scene option (`scene_option`): 
    - 1: Single rectangular static obstacle 
        - (1-small, 2-medium, 3-large)
    - 2: Two rectangular static obstacles 
        - (1-small stagger, 2-large stagger, 3-close aligned, 4-far aligned)
    - 3: Single non-convex static obstacle
        - (1-big u-shape, 2-small u-shape, 3-big v-shape, 4-small v-shape)
    - 4: Single dynamic obstacle
        - (1-crash, 2-cross)
    """
    scene_1_robot = MobileRobot(np.array([0.6, 3.5, 0, 0, 0]))
    scene_1_boundary = Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)])
    scene_1_obstacles_list = [[(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                            [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                            [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                            [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    scene_1_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_1_obstacles_list]
    scene_1_goal = Goal((15.4, 3.5))

    unexpected_obstacles:List[Obstacle] = []

    if sub_index == 1:
        if scene_option == 1:
            unexpected_obstacle = Obstacle.create_mpc_static([(7.5, 3.0), (7.5, 4.0), (8.5, 4.0), (8.5, 3.0)]) # small
        elif scene_option == 2:
            unexpected_obstacle = Obstacle.create_mpc_static([(7.2, 2.8), (7.2, 4.2), (8.8, 4.2), (8.8, 2.8)]) # medium
        elif scene_option == 3:
            unexpected_obstacle = Obstacle.create_mpc_static([(7.0, 2.5), (7.0, 4.5), (9.0, 4.5), (9.0, 2.5)]) # large
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~3.")
        unexpected_obstacle.visible_on_reference_path = False
        unexpected_obstacles.append(unexpected_obstacle)

    elif sub_index == 2:
        if scene_option == 1:
            unexpected_obstacle_1 = Obstacle.create_mpc_static([(5,1.5), (5,4), (6,4), (6,1.5)])
            unexpected_obstacle_2 = Obstacle.create_mpc_static([(8.5, 3.5), (8.5, 8.0), (9.5, 8.0), (9.5, 3.5)])
        elif scene_option == 2:
            unexpected_obstacle_1 = Obstacle.create_mpc_static([(5,1.5), (5,5), (6,5), (6,1.5)])
            unexpected_obstacle_2 = Obstacle.create_mpc_static([(8.5, 3.5), (8.5, 8.0), (9.5, 8.0), (9.5, 3.5)])
        elif scene_option == 3:
            unexpected_obstacle_1 = Obstacle.create_mpc_static([(4.2, 2.8), (4.2, 4.2), (5.8, 4.2), (5.8, 2.8)])
            unexpected_obstacle_2 = Obstacle.create_mpc_static([(6.2, 2.8), (6.2, 4.2), (7.8, 4.2), (7.8, 2.8)])
        elif scene_option == 4:
            unexpected_obstacle_1 = Obstacle.create_mpc_static([(4.2, 2.8), (4.2, 4.2), (5.8, 4.2), (5.8, 2.8)])
            unexpected_obstacle_2 = Obstacle.create_mpc_static([(8.2, 2.8), (8.2, 4.2), (9.8, 4.2), (9.8, 2.8)])
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~4.")
        unexpected_obstacles.append(unexpected_obstacle_1)
        unexpected_obstacles.append(unexpected_obstacle_2)

    elif sub_index == 3:
        unexpected_obstacle_3 = None
        if scene_option == 1:
            unexpected_obstacle_1 = Obstacle.create_mpc_static([(6.0, 4.5), (6.0, 5.0), (8.5, 5.0), (8.5, 4.5)])
            unexpected_obstacle_2 = Obstacle.create_mpc_static([(8.5, 5.0), (8.5, 2.0), (8.0, 2.0), (8.0, 5.0)])
            unexpected_obstacle_3 = Obstacle.create_mpc_static([(8.5, 2.0), (6.0, 2.0), (6.0, 2.5), (8.5, 2.5)])
        elif scene_option == 2:
            unexpected_obstacle_1 = Obstacle.create_mpc_static([(6.0, 4.0), (6.0, 4.5), (7.5, 4.5), (7.5, 4.0)])
            unexpected_obstacle_2 = Obstacle.create_mpc_static([(7.5, 4.5), (7.5, 2.0), (7.0, 2.0), (7.0, 4.5)])
            unexpected_obstacle_3 = Obstacle.create_mpc_static([(7.5, 2.0), (6.0, 2.0), (6.0, 2.5), (7.5, 2.5)])
        elif scene_option == 3:
            unexpected_obstacle_1 = Obstacle.create_mpc_static([(6.0, 5.0), (9.5, 5.0), (9.5, 3.5), (9.0, 3.5)])
            unexpected_obstacle_2 = Obstacle.create_mpc_static([(9.5, 3.5), (9.5, 2.0), (6.0, 2.0), (9.0 ,3.5)])
        elif scene_option == 4:
            unexpected_obstacle_1 = Obstacle.create_mpc_static([(6.5, 4.5), (8.5, 4.5), (8.5, 3.5), (8.0, 3.5)])
            unexpected_obstacle_2 = Obstacle.create_mpc_static([(8.5, 3.5), (8.5, 2.5), (6.5, 2.5), (8.0, 3.5)])
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~4.")
        unexpected_obstacles.append(unexpected_obstacle_1)
        unexpected_obstacles.append(unexpected_obstacle_2)
        if unexpected_obstacle_3 is not None:
            unexpected_obstacles.append(unexpected_obstacle_3)

    elif sub_index == 4:
        if scene_option == 1:
            unexpected_obstacle = Obstacle.create_mpc_dynamic(p1=(15.4, 3.5), p2=(0.6, 3.5), freq=0.15, rx=0.8, ry=0.8, angle=0.0, corners=20)
            unexpected_obstacles.append(unexpected_obstacle)
        elif scene_option == 2:
            unexpected_obstacle = Obstacle.create_mpc_dynamic(p1=(10.0, 1.0), p2=(10.0, 9.0), freq=0.2, rx=0.8, ry=0.8, angle=0.0, corners=20)
            unexpected_obstacles.append(unexpected_obstacle)
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~2.")
    
    else:
        raise ValueError(f"Invalid scene index, should be 1~4.")

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False
    scene_1_obstacles.extend(unexpected_obstacles)

    return scene_1_robot, scene_1_boundary, scene_1_obstacles, scene_1_goal


def generate_map_scene_2(sub_index: int, scene_option: int) -> MapDescription:
    """
    Subscene index (`sub_index`) with scene option (`scene_option`): 
    - 1: Single rectangular obstacle
        - (1-right, 2-sharp, 3-u-shape)
    - 2: Single dynamic obstacle
        - (1-right, 2-sharp, 3-u-shape)
    """
    scene_2_robot = MobileRobot(np.array([3, 0.6, np.math.pi/2, 0, 0]))
    scene_2_boundary = Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 18.0), (0.0, 18.0)])
    scene_2_obstacles_list = [[(0.0, 0.0), (0.0, 16.0), (1.0, 16.0), (1.0, 0.0)],]
    scene_2_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_2_obstacles_list]
    scene_2_goal_1 = Goal((15.5, 14.0))
    scene_2_goal_2 = Goal((11.0, 0.6))
    scene_2_goal_3 = Goal((5.5, 0.6))

    more_obstacles:List[Obstacle] = []
    unexpected_obstacles:List[Obstacle] = []

    if sub_index == 1:
        if scene_option == 1:
            scene_2_goal = scene_2_goal_1
            more_obstacle_1 = Obstacle.create_mpc_static([(4.0, 0.0), (16.0, 0.0), (16.0, 13.0), (4.0, 13.0)]) # right
            more_obstacle_2 = None
            unexpected_obstacle = Obstacle.create_mpc_static([(3.0, 14.0), (4.0, 14.0), (4.0, 13.0), (3.0, 13.0)])
        elif scene_option == 2:
            scene_2_goal = scene_2_goal_2
            more_obstacle_1 = Obstacle.create_mpc_static([(4.0, 0.0), (4.0, 13.0), (4.5, 13.0), (10.0, 0.0)]) # sharp
            more_obstacle_2 = Obstacle.create_mpc_static([(15.0, 0.0), (16.0, 0.0), (16.0, 16.0), (8.0, 16.0)]) # sharp
            unexpected_obstacle = Obstacle.create_mpc_static([(4.0, 13.5), (4.0, 14.0), (4.5, 14.0), (4.5, 13.5)])
        elif scene_option == 3:
            scene_2_goal = scene_2_goal_3
            more_obstacle_1 = Obstacle.create_mpc_static([(4.0, 0.0), (4.0, 13.0), (4.5, 13.0), (4.5, 0.0)]) # u-turn
            more_obstacle_2 = Obstacle.create_mpc_static([(7.5, 0.0), (16.0, 0.0), (16.0, 16.0), (7.5, 16.0)]) # u-turn
            unexpected_obstacle = Obstacle.create_mpc_static([(4.0, 13.5), (4.0, 14.0), (4.5, 14.0), (4.5, 13.5)])
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~3.")
        more_obstacles.append(more_obstacle_1)
        if more_obstacle_2 is not None:
            more_obstacles.append(more_obstacle_2)
        unexpected_obstacles.append(unexpected_obstacle)

    elif sub_index == 2:
        raise NotImplementedError
        if scene_option == 1:
            scene_2_goal = scene_2_goal_2
            more_obstacle_1 = Obstacle.create_mpc_static([(4.0, 0.0), (4.0, 13.0), (4.5, 13.0), (10.0, 0.0)]) # sharp
            more_obstacle_2 = Obstacle.create_mpc_static([(15.0, 0.0), (16.0, 0.0), (16.0, 16.0), (8.0, 16.0)]) # sharp
            unexpected_obstacle = Obstacle.create_mpc_static([(4.0, 13.5), (4.0, 14.0), (4.5, 14.0), (4.5, 13.5)])
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1.")
        more_obstacles.append(more_obstacle_1)
        more_obstacles.append(more_obstacle_2)
        unexpected_obstacles.append(unexpected_obstacle)
    
    else:
        raise ValueError(f"Invalid scene index, should be 1~4.")

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False
    scene_2_obstacles.extend(unexpected_obstacles)
    scene_2_obstacles.extend(more_obstacles)

    return scene_2_robot, scene_2_boundary, scene_2_obstacles, scene_2_goal









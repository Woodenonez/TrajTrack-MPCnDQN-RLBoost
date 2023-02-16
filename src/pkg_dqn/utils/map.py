"""
This file is used to generate the maps which the DRL agent will be trained and
evaluated in. Such a map constitutes e.g. the initial robot position, the goal
position and the locations of obstacles and boundaries.
"""

import random
from math import pi, radians, cos, sin

import numpy as np
import shapely.ops
from shapely.geometry import LineString, Polygon, JOIN_STYLE, Point

from ..environment import MobileRobot, Obstacle, Boundary, Goal, MapDescription, MapGenerator

from typing import Union


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
            'start': (1, 1, radians(0)),
            'goal': (8, 8, radians(90)),
        },
        {
            'boundary': [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)],
            'static_obstacles': [
                [(5.0, 0.0), (5.0, 15.0), (7.0, 15.0), (7.0, 0.0)],
                [(12.0, 12.5), (12.0, 20.0), (15.0, 20.0), (15.0, 12.5)], 
                [(12.0, 0.0), (12.0, 7.5), (15.0, 7.5), (15.0, 0.0)]],
            'start': (1, 5, radians(45)),
            'goal': (19, 10, radians(0)),
        },
        {
            'boundary': [(0.0, 0.0), (10.0, 0.0), (10.0 ,10.0), (25.0, 10.0), (25.0, 0.0), (50.0, 0), (50, 50), (0, 50), (0, 16), (10, 16), (10, 45), (15, 45), (15, 30), (35, 30), (35, 15), (0, 15)],
            'static_obstacles': [
                [(30.0, 5.0), (30.0, 14.5), (40.0, 14.5), (40, 5.0)],
                [(45.0, 15.0), (44.0, 20.0), (46.0, 20.0)],
                [(25, 35), (25, 40), (40, 40), (40, 35)],
                [(32.0, 6.0), (32.0, 10.5), (42.0, 12.5), (42, 8.0)]],
            'start': (1, 1, radians(225)),
            'goal': (5, 20, radians(270)),
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
            'start': (30, 5, radians(90)),
            'goal': (30, 55, radians(90)),
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
            'start': (10.3, 55.8, radians(270)),
            'goal': (38.1, 25.0, radians(300)),
        },
        {
            'boundary': [(0.37, 0.32), (5.79, 0.31), (5.79, 5.18), (0.14, 5.26)],
            'static_obstacles': [[(2.04, 0.28), (2.0, 3.8), (2.8, 3.81), (2.78, 0.29)]],
            'start': (1.01, 0.98, radians(90)),
            'goal': (3.82, 1.05, radians(270)),
        },
        {
            'boundary': [(1.55, 1.15), (29.0, 1.1), (29.0, 28.75), (0.85, 28.9), (0.85, 1.15)],
            'static_obstacles': [[(5.6, 3.3), (5.75, 20.15), (18.35, 20.05), (18.35, 19.7), (7.25, 19.7), (7.05, 3.2)], [(13.85, 23.4), (21.25, 23.35), (21.1, 16.4), (6.9, 16.35), (6.7, 12.9), (23.45, 13.25), (23.4, 25.05), (13.0, 25.35)]],
            'start': (2.95, 13.5, radians(90)),
            'goal': (9.6, 18.1, radians(180)),
        },
        {
            'boundary': [(2.0, 1.08), (22.8, 1.12), (22.84, 19.16), (1.8, 19.24)],
            'static_obstacles': [[(9.64, 5.28), (9.56, 10.72), (8.68, 11.88), (9.48, 12.2), (10.52, 10.96), (11.6, 12.12), (12.6, 11.36), (11.28, 10.4), (11.6, 0.56), (9.68, 0.68)]],
            'start': (7.16, 8.16, radians(90)),
            'goal': (12.72, 9.32, radians(265)),
        },
        {
            'boundary': [(0.96, 1.88), (22.88, 1.72), (22.92, 20.8), (0.64, 20.92)],
            'static_obstacles': [[(9.12, 1.48), (8.8, 9.56), (9.76, 12.72), (10.8, 9.56), (11.08, 1.48)]],
            'start': (7.44, 6.16, radians(90)),
            'goal': (12.44, 6.4, radians(265)),
        },
        {
            'boundary': [(2.36, 1.6), (22.6, 1.84), (22.16, 21.04), (1.52, 20.88)],
            'static_obstacles': [[(9.92, 1.24), (9.64, 8.52), (12.6, 10.44), (15.6, 8.76), (15.76, 1.08)]],
            'start': (7.08, 5.88, radians(90)),
            'goal': (17.8, 6.56, radians(265)),
        },
        {
            'boundary': [(1.5, 1.0), (1.7, 58.6), (59.0, 58.4), (58.6, 1.3)],
            'static_obstacles': [
                [(27.0, 6.0), (27.0, 33.0), (4.0, 33.0), (4.0, 6.0)], 
                [(65.0, 6.0), (28.1, 6.0), (28.1, 33.0), (65.0, 33.0)], 
                [(4.4, 34.1), (44.0, 34.1), (44.0, 39.3), (55.3, 39.6), (55.3, 42.8), (44.0, 42.3), (44.1, 49.1), (54.9, 49.2), (54.9, 53.0), (4.7, 53.0)], 
                [(47.7, 36.2), (47.7, 34.6), (57.8, 34.5), (57.8, 36.3)]],
            'start': (27.8, 2.7, radians(90)),
            'goal': (50.3, 45.9, radians(0)),
        },
        {
            'boundary': [(11.9, 3.6), (11.9, 50.6), (47.3, 50.6), (47.3, 3.6)],
            'static_obstacles': [
                [(11.9, 11.8), (22.2, 11.8), (22.2, 15.9), (11.9, 15.9)],
                [(11.9, 20.4), (22.2, 20.4), (22.2, 25.0), (11.9, 25.0)],
                [(28.0, 25.5), (28.0, 20.5), (32.4, 20.5), (32.4, 15.7), (28.0, 15.7), (28.0, 3.6), (37.8, 3.6), (37.8, 25.5)], 
                [(15.9, 29.5), (37.7, 29.5), (37.7, 44.5), (25.3, 44.5), (25.3, 40.7), (35.0, 40.7), (35.0, 31.7), (15.9, 31.7)],
                [(29.8, 28.7), (29.8, 25.8), (34.5, 25.8), (34.5, 28.7)]],
            'start': (18.9, 7.0, radians(45)),
            'goal': (44.7, 6.8, radians(270)),
            'dynamic_obstacles': [
                [[18.5, 18.2],[28.1, 18.2], 0.06, 0.5, 1.0, pi / 2],
                [[16.775, 34.0], [22.5, 42.2], 0.07, 0.3, 0.7, pi / 2 + 0.961299],
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

    init_state = np.array([5, random.uniform(5, 15), random.uniform(0, 2*pi), 0, 0])
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
            angle = random.uniform(0, 2 * pi)

            obstacles.append(Obstacle.create_mpc_dynamic((x, y), (x2, y2), freq, rx, ry, angle))
    goal = Goal((35, random.uniform(5, 15)))

    return robot, boundary, obstacles, goal


def generate_map_corridor() -> MapDescription:
    """
    Generates a map with a randomized narrow corridor ando no dynamic obstacles.
    """

    max_angle = pi / 2

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
        coords = np.vstack((coords, coords[i + 1, :] + length * np.asarray((cos(angle), sin(angle)))))
    coords = np.vstack((coords, coords[-1, :] + (wall_padding, 0)))

    init_state = np.array([coords[0, 0], coords[0, 1], random.uniform(0, 2 * pi), 0, 0])
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

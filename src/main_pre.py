import random

import numpy as np
from shapely.geometry import Polygon, Point, LineString

from pkg_map.map_geometric import GeometricMap
from pkg_obstacle import geometry_tools

from pkg_dqn.environment import MapDescription
from pkg_dqn.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc
from pkg_dqn.utils.map import generate_map_scene_1

from typing import List, Tuple


class Inflator:
    def __init__(self, inflate_margin):
        self.inflate_margin = inflate_margin

    def __call__(self, polygon: List[tuple]):
        shapely_inflated = geometry_tools.polygon_inflate(Polygon(polygon), self.inflate_margin)
        return geometry_tools.polygon_to_vertices(shapely_inflated)
    

class HintSwitcher:
    def __init__(self, switch_distance: float):
        self.switch_distance = switch_distance
        self.switch_on = False

    def __call__(self, pred_positions: List[tuple], obstacle_list: List[List[tuple]]) -> bool:
        self.switch_on = False
        current_position = pred_positions[0]
        for pred_position in pred_positions:
            for obstacle in obstacle_list:
                shapely_obstacle = Polygon(obstacle)
                if shapely_obstacle.contains(Point(pred_position)):
                    if shapely_obstacle.distance(Point(current_position)) < self.switch_distance:
                        self.switch_on = True
        return self.switch_on


def get_geometric_map(rl_map: MapDescription, inflate_margin: float) -> GeometricMap:
    _, rl_boundary, rl_obstacles, _ = rl_map
    inflator = Inflator(inflate_margin)
    geometric_map = GeometricMap(
        boundary_coords=rl_boundary.vertices.tolist(),
        obstacle_list=[obs.nodes.tolist() for obs in rl_obstacles if obs.is_static],
        inflator=inflator
    )
    return geometric_map

def generate_map() -> MapDescription:
    """
    MapDescription = Tuple[MobileRobot, Boundary, List[Obstacle], Goal]
    """
    return generate_map_scene_1(1, 2)
    # return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc()])()
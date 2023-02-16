import random

import numpy as np
from shapely.geometry import Polygon

from pkg_map.map_geometric import GeometricMap
from pkg_obstacle import geometry_tools

from pkg_dqn.environment import MapDescription
from pkg_dqn.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc

from typing import List, Tuple


class Inflator:
    def __init__(self, inflate_margin):
        self.inflate_margin = inflate_margin

    def __call__(self, polygon: List[tuple]):
        shapely_inflated = geometry_tools.polygon_inflate(Polygon(polygon), self.inflate_margin)
        return geometry_tools.polygon_to_vertices(shapely_inflated)


def get_geometric_map(rl_map: MapDescription, inflate_margin: float):
    _, rl_boundary, rl_obstacles, _ = rl_map
    inflator = Inflator(inflate_margin)
    geometric_map = GeometricMap(
        boundary_coords=rl_boundary.vertices.tolist(),
        obstacle_list=[obs.nodes.tolist() for obs in rl_obstacles],
        inflator=inflator
    )
    return geometric_map

def generate_map() -> MapDescription:
    """
    MapDescription = Tuple[MobileRobot, Boundary, List[Obstacle], Goal]
    """
    return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc()])()
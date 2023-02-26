from gym import spaces
import numpy as np
import numpy.typing as npt
from math import pi, cos, sin
from shapely.geometry import Polygon, LineString
from matplotlib.axes import Axes
from . import Component
from .utils import normalize_distance
from .. import plot


class SectorObservation(Component):
    """
    Divides the space around the robot and reports the distance to the closest
    obstacle within each sector
    """
    def __init__(self, num_segments: int) -> None:
        """
        :param num_segments: The number of sectors to use
        """
        self.num_segments = num_segments
        self.external_obs_space = spaces.Box(0, 1, shape=(self.num_segments,), dtype=np.float32)

    def external_obs(self) -> npt.ArrayLike:
        L = 1000

        obs = np.zeros((self.num_segments,), dtype=np.float32)
        self.distances = np.zeros((self.num_segments,), dtype=np.float32)

        geometries = [o.padded_polygon for o in self.env.obstacles]
        geometries.append(LineString(self.env.boundary.padded_polygon.exterior.coords))

        segment_width = 2 * pi / self.num_segments
        for i in range(self.num_segments):
            angle1 = self.env.agent.angle + (i - 1 / 2) * segment_width
            angle2 = angle1 + segment_width
            points = [(0, 0), (L * cos(angle1), L * sin(angle1)), (L * cos(angle2), L*sin(angle2))]
            segment = Polygon(self.env.agent.position + points)

            closest_distance = float("inf")
            for geometry in geometries:
                intersecion = segment.intersection(geometry)
                if not intersecion.is_empty:
                    distance = intersecion.distance(self.env.agent.point)
                    if distance < closest_distance:
                        closest_distance = distance
            
            self.distances[i] = closest_distance
            obs[i] = normalize_distance(closest_distance)

        return obs

    def render(self, ax: Axes) -> None:
        plot.sectors(ax, self.distances, robot=self.env.agent)
    

from gym import spaces
import numpy as np
import numpy.typing as npt
from math import pi, cos, sin
from shapely.geometry import Polygon, LineString
from matplotlib.axes import Axes
from . import Component
from .utils import normalize_distance
from .. import plot


class SectorAndRayObservation(Component):
    """
    The combination of ``SectorObservation`` and ``RayObservation``, see the
    accompanying report for details
    """
    def __init__(self, num_segments: int, use_memory: bool) -> None:
        """
        :param num_segments: The number of sectors/rays to use
        :param use_memory: Whether to also observe quantities from the previous
                           time step in order to aid dynamic obstacle avoidance
        """
        self.num_segments = num_segments
        self.use_memory = use_memory
        if use_memory:
            self.external_obs_space = spaces.Box(0, 1, shape=(self.num_segments * 4,), dtype=np.float32)
            self.old_obs = np.zeros((self.num_segments * 4,), dtype=np.float32)
        else:
            self.external_obs_space = spaces.Box(0, 1, shape=(self.num_segments * 2,), dtype=np.float32)    

    def external_obs(self) -> npt.ArrayLike:
        L = 1000
        if self.use_memory:
            obs = np.zeros((self.num_segments * 4,), dtype=np.float32)
        else:
            obs = np.zeros((self.num_segments * 2,), dtype=np.float32)

        self.segments = np.zeros((self.num_segments,), dtype=np.float32)
        self.rays = np.zeros((self.num_segments,), dtype=np.float32)

        geometries = [o.padded_polygon for o in self.env.obstacles]
        geometries.append(LineString(self.env.boundary.padded_polygon.exterior.coords))

        segment_width = 2 * pi / self.num_segments
        for i in range(self.num_segments):
            angle = self.env.agent.angle + i * segment_width

            angle1 = angle - segment_width / 2
            angle2 = angle + segment_width / 2

            ray = LineString(self.env.agent.position + [(0, 0), (L * cos(angle), L * sin(angle))])

            points = [(0, 0), (L * cos(angle1), L * sin(angle1)), (L * cos(angle2), L*sin(angle2))]
            segment = Polygon(self.env.agent.position + points)

            closest_in_segment = float("inf")
            closest_on_ray = float("inf")
            for geometry in geometries:
                intersecion = segment.intersection(geometry)
                if not intersecion.is_empty:
                    closest_in_segment = min(closest_in_segment, intersecion.distance(self.env.agent.point))
                    
                    intersecion = intersecion.intersection(ray)
                    if not intersecion.is_empty:
                        closest_on_ray = min(closest_on_ray, intersecion.distance(self.env.agent.point))
            
            self.segments[i] = closest_in_segment
            self.rays[i] = closest_on_ray

        if self.use_memory:
            obs[:self.num_segments] = normalize_distance(self.segments)
            obs[self.num_segments:2*self.num_segments] = normalize_distance(self.rays)
            obs[2*self.num_segments:3*self.num_segments] = self.old_obs[:self.num_segments]
            obs[3*self.num_segments:] = self.old_obs[self.num_segments:2*self.num_segments]
            
            self.old_obs = obs
        else:
            obs[:self.num_segments] = normalize_distance(self.segments)
            obs[self.num_segments:] = normalize_distance(self.rays)
        
        return obs

    def render(self, ax: Axes) -> None:
        plot.sectors(ax, self.segments, robot=self.env.agent)
        plot.rays(ax, self.rays, robot=self.env.agent)

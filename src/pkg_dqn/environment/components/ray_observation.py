from gym import spaces
import numpy as np
import numpy.typing as npt
from shapely.geometry import LineString
from matplotlib.axes import Axes
from . import Component
from .utils import normalize_distance
from .. import plot


class RayObservation(Component):
    """
    Casts rays from the robot against all obstacles in the environment and
    observes the length of these rays
    """
    def __init__(self, num_rays: int) -> None:
        """
        :param num_rays: The number of rays to use
        """
        self.num_rays = num_rays
        self.external_obs_space = spaces.Box(0, 1, shape=(self.num_rays,), dtype=np.float32)

    def external_obs(self) -> npt.ArrayLike:
        L = 1000

        obs = np.zeros((self.num_rays,), dtype=np.float32)
        self.distances = np.zeros((self.num_rays,), dtype=np.float32)

        geometries = [o.padded_polygon for o in self.env.obstacles]
        geometries.append(LineString(self.env.boundary.padded_polygon.exterior.coords))

        segment_width = 2 * np.pi / self.num_rays
        for i in range(self.num_rays):
            angle = self.env.agent.angle + i * segment_width
            ray = LineString(self.env.agent.position + [(0, 0), (L*np.cos(angle), L*np.sin(angle))])

            closest_distance = float("inf")
            for geometry in geometries:
                intersecion = ray.intersection(geometry)
                if not intersecion.is_empty:
                    distance = intersecion.distance(self.env.agent.point)
                    if distance < closest_distance:
                        closest_distance = distance
            
            self.distances[i] = closest_distance
            obs[i] = normalize_distance(closest_distance)

        return obs

    def render(self, ax: Axes) -> None:
        plot.rays(ax, self.distances, robot=self.env.agent)

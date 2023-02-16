import numpy as np
from shapely.geometry.base import BaseGeometry

from ._obstacle import Obstacle, MotionModel
from .geometry_shapely import ShapelyPolygon, ShapelyCircle, ShapelyEllipse

from . import geometry_tools

from typing import Callable, Any, Tuple
from matplotlib.axes import Axes

"""
All obstacles are based on the Shapely geometry objects.
"""


class ObstacleTemplate(Obstacle):
    def __init__(self, geometry: BaseGeometry, geometry_shape: str, motion_model:MotionModel=None, id_:int=None, name:str=None) -> None:
        super().__init__(geometry, geometry_shape, motion_model, id_, name)
        self.geometry: BaseGeometry

    @property
    def state(self) -> np.ndarray:
        x, y, theta = 0.0, 0.0, 0.0
        return np.array([x, y, theta])

    def plot(self, ax: Axes, **kwargs) -> None:
        pass


class PolygonObstacle(ObstacleTemplate):
    def __init__(self, geometry: ShapelyPolygon, motion_model:MotionModel=None, id_:int=None, name:str=None) -> None:
        super().__init__(geometry, "polygon", motion_model, id_, name)
        self.geometry: ShapelyPolygon

    @property
    def position(self) -> tuple:
        """Return the position of the obstacle."""
        return self.geometry.centroid.coords[0]

    @property
    def state(self) -> np.ndarray:
        x, y = self.position
        return np.array([x, y, self.geometry.angle])

    def step(self, action: Any, dt:float=None) -> None:
        """Step the obstacle forward in time."""
        if self.motion_model is not None:
            new_state = self.motion_model(self.state, action, dt)
            d_x = new_state[0] - self.state[0]
            d_y = new_state[1] - self.state[1]
            d_angle = new_state[2] - self.state[2]
            shapely_geo = geometry_tools.geometry_translate(self.geometry.to_shapely(), (d_x, d_y))
            shapely_geo = geometry_tools.geometry_rotate(shapely_geo, d_angle, origin='centroid')
            self.geometry = ShapelyPolygon.from_shapely(shapely_geo, angle=(new_state[2]%(2*np.pi)))

    def plot(self, ax: Axes, **kwargs) -> None:
        """Plot the obstacle on the given axes."""
        self.geometry.plot(ax, **kwargs)


class EllipseObstacle(ObstacleTemplate):
    def __init__(self, geometry: ShapelyEllipse, motion_model:MotionModel=None, id_:int=None, name:str=None) -> None:
        super().__init__(geometry, "ellipse", motion_model, id_, name)
        self.geometry: ShapelyEllipse

    @property
    def position(self) -> tuple:
        """Return the position of the obstacle."""
        return self.geometry.center

    @property
    def state(self) -> np.ndarray:
        x, y = self.position
        return np.array([x, y, self.geometry.angle])

    def step(self, action: Any, dt:float=None) -> None:
        """Step the obstacle forward in time."""
        if self.motion_model is not None:
            new_state = self.motion_model(self.state, action, dt)
            new_center = tuple(new_state[:2])
            new_angle = new_state[2]
            self.geometry = ShapelyEllipse(new_center, self.geometry.radii, new_angle, self.geometry._n_approx)

    def plot(self, ax: Axes, approx:bool=True, **kwargs) -> None:
        """Plot the obstacle on the given axes."""
        self.geometry.plot(ax, approx, **kwargs)
        direction_point = (self.geometry.center[0] + self.geometry.radii[0]*np.cos(self.geometry.angle),
                           self.geometry.center[1] + self.geometry.radii[0]*np.sin(self.geometry.angle))
        ax.arrow(self.geometry.center[0], self.geometry.center[1], 
                 direction_point[0] - self.geometry.center[0], direction_point[1] - self.geometry.center[1], 
                 head_width=0.1, head_length=0.1, fc='k', ec='k')


class CircleObstacle(ObstacleTemplate):
    def __init__(self, geometry: ShapelyCircle, motion_model:MotionModel=None, id_:int=None, name:str=None) -> None:
        super().__init__(geometry, "circle", motion_model, id_, name)
        self.geometry: ShapelyCircle

    @property
    def position(self) -> tuple:
        """Return the position of the obstacle."""
        return self.geometry.center

    @property
    def state(self) -> np.ndarray:
        x, y = self.position
        return np.array([x, y, self.geometry.angle])

    def step(self, action: Any, dt:float=None) -> None:
        """Step the obstacle forward in time."""
        if self.motion_model is not None:
            new_state = self.motion_model(self.state, action, dt)
            new_center = tuple(new_state[:2])
            new_angle = new_state[2]
            self.geometry = ShapelyCircle(new_center, self.geometry.radius, new_angle, self.geometry._n_approx)

    def plot(self, ax: Axes, approx:bool=True, **kwargs) -> None:
        """Plot the obstacle on the given axes."""
        self.geometry.plot(ax, approx, **kwargs)
        direction_point = (self.geometry.center[0] + self.geometry.radius*np.cos(self.geometry.angle),
                           self.geometry.center[1] + self.geometry.radius*np.sin(self.geometry.angle))
        ax.arrow(self.geometry.center[0], self.geometry.center[1], 
                 direction_point[0] - self.geometry.center[0], direction_point[1] - self.geometry.center[1], 
                 head_width=0.1, head_length=0.1, fc='k', ec='k')



"""
Util function for plotting vareous objects
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString

from . import Obstacle, Boundary, MobileRobot, Goal

from typing import Iterable, Union
import numpy.typing as npt
from matplotlib.axes import Axes

def line(ax: Axes, l: Union[LineString, npt.ArrayLike], *args, **kwargs) -> None:
    coords = np.asarray(l.coords if isinstance(l, LineString) else l)
    ax.plot(coords[:, 0], coords[:, 1], *args, **kwargs)

def goal(ax: Axes, goal: Goal, fmt: str = 'y*', *args, **kwargs) -> None:
    ax.plot(goal.position[0], goal.position[1], fmt, *args, **kwargs)

def reference_path(ax: Axes, l: LineString, fmt: str = '--k', *args, label: str = 'Reference path', **kwargs) -> None:
    line(ax, l, fmt, *args, label=label, **kwargs)
    goal(ax, Goal(np.asarray(l.coords)[-1, :]), *args, markersize=15, label=None if label is None else 'Goal', **kwargs)

def polygon(ax: Axes, p: Union[Polygon, npt.ArrayLike], *args, **kwargs) -> None:
    p = np.asarray(list(p.exterior.coords) if isinstance(p, Polygon) else p, dtype=np.float32)
    p = np.vstack([p, p[0]])
    line(ax, p, *args, **kwargs)

def obstacle(ax: Axes, o: Obstacle, *args, padded: bool = False, **kwargs) -> None:
    polygon(ax, o.get_padded_vertices() if padded else o.get_vertices(), *args, **kwargs)

def obstacles(ax: Axes, obstacles: Iterable[Obstacle], fmt: str = 'r', *args, padded: bool = False, label: str = 'Obstacles', **kwargs) -> None:
    if len(obstacles) > 0:
        for o in obstacles:
            obstacle(ax, o, fmt, *args, padded=padded, **kwargs)
        obstacle(ax, o, fmt, *args, label=label, padded=padded, **kwargs)

def boundary(ax: Axes, b: Boundary, fmt: str = 'k', *args, padded: bool = False, label: str = 'Boundary', **kwargs) -> None:
    polygon(ax, b.get_padded_vertices() if padded else b.vertices, fmt, *args, label=label, **kwargs)

def robot(ax: Axes, robot: MobileRobot) -> None:
    n = 3
    X = np.zeros((2, n), dtype=float)
    for i, angle in enumerate([0, np.pi * 3 / 4, -np.pi * 3/4]):
        X[:, i] = (np.cos(angle), np.sin(angle))
    c = np.cos(robot.angle)
    s = np.sin(robot.angle)
    X = 0.9 * robot.cfg.RADIUS * np.array([[c, -s], [s, c]]) @ X

    ax.add_artist(plt.Circle(robot.position, robot.cfg.RADIUS, color="r", alpha=1, zorder=100))
    ax.add_artist(plt.Polygon((robot.position[:, None] + X).T, color="lightsalmon", zorder=101))

def sectors(ax: Axes, sectors: npt.NDArray, *args, angle: Union[float, None] = None, position: Union[npt.NDArray, None] = None, robot: Union[MobileRobot, None] = None, color='y', alpha: float = 0.5, label: str = None, **kwargs) -> None:
    if robot is not None:
        angle = robot.angle
        position = robot.position

    segment_width = 2 * np.pi / len(sectors)
    for i in range(len(sectors)):
        angle1 = angle + (i - 1 / 2) * segment_width
        angle2 = angle1 + segment_width

        N = 8
        xy = np.zeros((2, N + 2))
        xy[:, 0] = (0, 0)
        for j in range(N + 1):
            a = angle1 + j * (angle2 - angle1) / N
            xy[:, j + 1] = (np.cos(a), np.sin(a))
        xy = position[:, None] + sectors[i] * xy

        ax.fill(xy[0, :], xy[1, :], *args, color=color, alpha=alpha, label=label if i == 0 else None, **kwargs)

def rays(ax: Axes, rays: npt.NDArray, *args, angle: Union[float, None] = None, position: Union[npt.NDArray, None] = None, robot: Union[MobileRobot, None] = None, color='yellowgreen', alpha: float = 1, label: str = None, **kwargs) -> None:
    if robot is not None:
        angle = robot.angle
        position = robot.position
    
    segment_width = 2 * np.pi / len(rays)
    for i in range(len(rays)):
        a = angle + i * segment_width
        xy = np.array([(0, 0), (np.cos(a), np.sin(a))])
        xy = position + rays[i] * xy
        ax.plot(xy[:, 0], xy[:, 1], *args, color=color, alpha=alpha, label=label if i == 0 else None, **kwargs)

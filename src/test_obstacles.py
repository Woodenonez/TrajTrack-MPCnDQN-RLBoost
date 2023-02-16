import sys
import math

from pkg_obstacle.geometry_shapely import ShapelyPolygon, ShapelyCircle, ShapelyEllipse
from pkg_obstacle.obstacle import ObstacleTemplate, CircleObstacle, EllipseObstacle, PolygonObstacle
from pkg_motion_model.motion_model import MotionModel
from pkg_motion_model.motion_model import OmnidirectionalModel, UnicycleModel

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Union


def test(geometry_list: Union[ShapelyCircle, ShapelyEllipse, ShapelyPolygon],
         motion_model_list: List[MotionModel], 
         action_list: List[np.ndarray],
         max_step:int=200,
         pause_interval:float=0.0):

    obstacle_list:List[ObstacleTemplate] = []
    for geometry, motion_model in zip(geometry_list, motion_model_list):
        if isinstance(geometry, ShapelyCircle):
            obstacle_list.append(CircleObstacle(geometry, motion_model=motion_model))
        elif isinstance(geometry, ShapelyEllipse):
            obstacle_list.append(EllipseObstacle(geometry, motion_model=motion_model))
        elif isinstance(geometry, ShapelyPolygon):
            obstacle_list.append(PolygonObstacle(geometry, motion_model=motion_model))
        else:
            raise TypeError('Unknown geometry type.')

    fig, ax = plt.subplots()
    for i in range(max_step):
        for obstacle, action in zip(obstacle_list, action_list):
            obstacle.step(action)
        if i in list(np.linspace(0, max_step, 20, dtype=int))+[0, max_step-1]:
            ax.cla()
            ax.axhline(0, color='black', ls='--', lw=1)
            ax.axvline(0, color='black', ls='--', lw=1)
            for obstacle in obstacle_list:
                obstacle.plot(ax)
            ax.axis('equal')
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)
            
            plt.pause(0.1)
            if pause_interval == 0.0:
                while not plt.waitforbuttonpress(0.1):
                    pass

    plt.show()


if __name__ == '__main__':
    ts = 0.1
    action1 = np.array([0.2, 0.2, np.pi/20])
    action2 = np.array([math.hypot(0.2, 0.2), np.pi/20])
    action3 = action2

    circle1 = ShapelyEllipse((0, 0), (2,1), 0, n=20)
    circle2 = ShapelyCircle((0, 0), 1, n=20)
    poly3 = ShapelyPolygon([(-1, -1), (-1, 1), (1, 1), (1, -1)])

    motion_model1 = OmnidirectionalModel(ts)
    motion_model2 = UnicycleModel(ts)
    motion_model3 = UnicycleModel(ts)

    test(geometry_list=[circle1, circle2, poly3], 
         motion_model_list=[motion_model1, motion_model2, motion_model3],
         action_list=[action1, action2, action3],
         pause_interval=0.0)

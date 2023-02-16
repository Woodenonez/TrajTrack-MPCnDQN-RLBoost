from gym import spaces
from typing import Union
import cv2
import numpy as np
import numpy.typing as npt
from .utils import normalize_distance

from . import Component


class ImageObservation(Component):
    """
    Observes images of the environment surrounding the robot, see the
    accompanying report for details
    """
    obs: Union[npt.NDArray[np.uint8], None] = None
    obstacles: list[list[npt.NDArray]] = []

    def __init__(self, width: int, height: int, scale_x: float, scale_y: float, down_sample: float, center_x: float, center_y: float, angle: float) -> None:
        """
        :param width: Final image width
        :param height: Final image height
        :param scale_x: The inverse of total width of the area depicted in the
                        image
        :param scale_y: The inverse of total height of the area depicted in the
                        image
        :param down_sample: The image is rendered at ``down_sample`` times
                            ``width`` by ``height``, and then down sampled to
                            ``width`` by ``height``
        :param center_x: The fraction (from 0 to 1) of the horizontal direction
                         where the robot should be located in the final image 
        :param center_y: The fraction (from 0 to 1) of the vertical direction
                         where the robot should be located in the final image
        :param angle: Rotation of the final image 
        """
        self.size = np.array([width, height])
        self.original_size = down_sample * self.size
        self.scale = np.array([scale_x, scale_y])
        self.center = np.array([center_x, center_y])
        self.angle = angle
        self.external_obs_space = spaces.Box(0, 255, shape=(3, self.size[1], self.size[0]), dtype=np.uint8)

        w = (self.size[0] - 1) / (scale_x * self.size[0])
        h = (self.size[1] - 1) / (scale_y * self.size[1])
        xrange = np.linspace(-w * center_x, w * (1 - center_x), self.size[0])
        yrange = np.linspace(-h * center_y, h * (1 - center_y), self.size[1])
        [x, y] = np.meshgrid(xrange, yrange)
        distance = normalize_distance(np.sqrt(x**2 + y**2))
        distance = distance - np.min(distance)
        self.distance_field = (255.5 * (1 - distance / np.max(distance))).astype(np.uint8)

    def reset(self) -> None:
        self.obs = None
        self.obstacles = []

    def external_obs(self) -> npt.ArrayLike:
        c = np.cos(self.env.agent.angle - self.angle)
        s = np.sin(self.env.agent.angle - self.angle)
        R = np.array([[s, -c], [c, s]])
        transform = lambda x: self.original_size * (self.scale * (R @ (x - self.env.agent.position).T).T + self.center)

        img0 = np.zeros((self.original_size[1], self.original_size[0]), dtype=np.uint8)
        cv2.fillPoly(img0, [np.int32(transform(self.env.boundary.get_padded_vertices()))], 255)
        img1 = img0.copy()

        n = 5
        self.obstacles = [[o.get_padded_vertices() for o in self.env.obstacles], *self.obstacles[0:n]]
        for obstacle in self.obstacles[0]:
            cv2.fillPoly(img0, [np.int32(transform(obstacle))], 0)
        for obstacle in self.obstacles[-1]:
            cv2.fillPoly(img1, [np.int32(transform(obstacle))], 0)

        img0 = cv2.resize(img0, self.size)
        img1 = cv2.resize(img1, self.size)

        self.obs = np.dstack((img0, img1, self.distance_field)).transpose([2,0,1])
        return self.obs

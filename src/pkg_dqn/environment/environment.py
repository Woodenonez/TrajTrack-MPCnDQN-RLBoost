from time import time
from packaging import version

import gym
from gym import spaces

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from shapely.geometry import LineString
from extremitypathfinder import PolygonEnvironment

from . import plot, MobileRobot, MapGenerator, MapDescription
from .components.component import Component

from typing import Union, List, Tuple
from numpy.typing import NDArray
from matplotlib.axes import Axes
from matplotlib.figure import Figure


GYM_0_22_X = version.parse(gym.__version__) >= version.parse("0.22.0")


class TrajectoryPlannerEnvironment(gym.Env):
    """
    Base class for the OpenAI gym environments.

    The observation space depends on the specific implementation, however the
    action space is always the same, see the accompanying report for details.
    The possible actions refer to angular and linear acceleration and are:
    0: accelerate left and accelerate forward
    1: keep anglular velocity and accelerate forward
    2: accelerate right and accelerate forward
    3: accelerate left and keep speed
    4: keep anglular velocity  and keep speed
    5: accelerate right and keep speed
    6: accelerate left and accelerate backward
    7: keep anglular velocity and accelerate backward
    8: accelerate right and accelerate backward
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    rendered:bool=False

    # Component producing external observations
    external_obs_component: Union[Component, None] = None

    def __init__(self, components: list[Component], generate_map: MapGenerator, time_step: float = 0.1):
        """
        :param components: The components which this environemnt should use.
        :param generate_map: Map generation function. 
        """
        self.components = components
        self.generate_map = generate_map
        self.time_step = time_step

        for component in self.components:
            component.env = self

            assert len(component.internal_obs_min) == len(component.internal_obs_max)
            if len(component.external_obs_space.shape) > 0:
                assert self.external_obs_component is None, "Environment can only have one external observation provider, this is a limitation which we did not have time to fix"

                self.external_obs_component = component

        # Observation space is determined from the components used
        self.observation_space = spaces.Dict(
            {
                "internal": spaces.Box(
                    np.hstack([np.asarray(c.internal_obs_min, dtype=np.float32) for c in self.components]),
                    np.hstack([np.asarray(c.internal_obs_max, dtype=np.float32) for c in self.components]),
                    dtype=np.float32
                ),
                **(
                    {} if self.external_obs_component is None else
                    {"external": self.external_obs_component.external_obs_space}
                )
            }
        )

        # Action space is always the same. Linear acceleration has three
        # possibilities, accelerate-cruise-deccelerate, and angular
        # acceleration also has three possibilities left-middle-right
        self.action_space = spaces.Discrete(3 * 3)

    def _get_info(self) -> dict:
        return {}

    def update_status(self, reset: bool = False) -> None:
        """
        Update some cached status variables for the current run of the
        environment.

        :param reset: Whether reset() was called on the environment
        """
        if reset:
            self.collided_with_obstacle = False
            self.collided_with_boundary = False
            self.collided = False
            self.reached_goal = False

            self.traversed_positions = [self.agent.position.copy()]
            self.speeds = [self.agent.speed]
            self.angular_velocities = [self.agent.angular_velocity]

        self.collided_with_obstacle |= any(o.collides(self.agent) for o in self.obstacles)
        self.collided_with_boundary |= self.boundary.collides(self.agent)
        self.collided |= self.collided_with_obstacle or self.collided_with_boundary
        self.reached_goal |= norm(self.goal.position - self.agent.position) < self.agent.cfg.RADIUS

        self.path_progress = self.path.project(self.agent.point)

        self.traversed_positions.append(self.agent.position.copy())
        self.speeds.append(self.agent.speed)
        self.angular_velocities.append(self.agent.angular_velocity)

    # @DeprecationWarning
    def _update_reference_path(self) -> bool:
        """
        Updates the reference path
        """

        # Only obstacles which should be visible on the reference path are
        # included in the reference path generation. Those obstacles that are
        # visible are padded by the agent radius
        obstacle_list_mitred = [
            o.get_mitred_vertices()
            for o in self.obstacles
            if o.visible_on_reference_path
        ]

        # Generate visibility grap
        environment = PolygonEnvironment()
        environment.store(
            self.boundary.get_mitred_vertices(), obstacle_list_mitred, validate=False
        )
        environment.prepare()

        # Find shortest path with A*
        path, _ = environment.find_shortest_path(self.agent.position, self.goal.position) # path: list[tuple[float, float]]
        self.path = LineString(path)
        return len(path) > 0

    def get_observation(self) -> dict:
        """Collects observations from all components and returns them"""
        self.obsv = {'internal': np.hstack([np.asarray(c.internal_obs(), dtype=np.float32) for c in self.components])}
        if self.external_obs_component is not None:
            self.obsv['external'] = self.external_obs_component.external_obs()
        return self.obsv
    
    def get_map_description(self) -> MapDescription:
        return (self.agent, self.boundary, self.obstacles, self.goal)

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        if GYM_0_22_X:
            super().reset(seed=seed) # following line to seed self.np_random

        while True:
            self.agent, self.boundary, self.obstacles, self.goal = self.generate_map()
            if self._update_reference_path():
                break
        self.update_status(True)

        self.last_render_at = 0

        for c in self.components:
            c.reset()

        observation = self.get_observation()
        info = self._get_info()

        if GYM_0_22_X:
            return observation, info
        else:
            return observation


    def set_agent_state(self, position: np.ndarray, angle: float, speed: float, angular_velocity: float) -> None:
        self.agent.position = position
        self.agent.angle = angle
        self.agent.speed = speed
        self.agent.angular_velocity = angular_velocity

    def set_reference_path(self, path: List[tuple]) -> None:
        self.path = LineString(path)

    def set_geometric_map(self, boundary, obstacle_list) -> None:
        raise NotImplementedError("This method is not implemented yet.")
        self.boundary = boundary
        self.obstacles = obstacle_list


    def step_obstacles(self) -> None:
        for obstacle in self.obstacles:
            obstacle.step(self.time_step)

    def step_agent(self, action: int) -> None:
        self.agent.step(action, self.time_step)

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self.step_obstacles()
        self.step_agent(action)

        self.update_status()

        observation = self.get_observation()
        reward = float(sum(c.step(action) for c in self.components))
        terminated = bool(self.collided or self.reached_goal)
        info = self._get_info()

        if GYM_0_22_X:
            return observation, reward, terminated, False, info
        else:
            return observation, reward, terminated, info

    def render(self, mode:str="human", pred_positions=None, ref_traj=None) -> Union[None, NDArray[np.uint8]]:
        if not self.rendered:
            self.rendered = True
            if mode == "human":
                plt.ion()
            self.fig, self.axes = plt.subplots(1, 2, constrained_layout=True)

        for ax in self.axes.ravel():
            ax.cla()

        plot.obstacles(self.axes[0], self.obstacles)
        plot.boundary(self.axes[0], self.boundary)
        plot.reference_path(self.axes[0], self.path)
        plot.robot(self.axes[0], self.agent)
        plot.line(self.axes[0], self.traversed_positions, "b", label="robot path")

        ### XXX
        if pred_positions is not None:
            self.axes[0].plot(np.array(pred_positions)[:, 0], np.array(pred_positions)[:, 1], "mx-", label="Predicted path")
        if ref_traj is not None:
            self.axes[0].plot(np.array(ref_traj)[:, 0], np.array(ref_traj)[:, 1], "gx-", label="Reference path")


        for component in self.components:
            component.render(self.axes[0])

        self.axes[0].legend(bbox_to_anchor=(0.5, 1.04), loc="lower center")
        self.axes[0].set_aspect('equal')

        times = np.arange(len(self.speeds))
        self.axes[1].plot(times, self.speeds, label="Speed [m/s]")
        self.axes[1].plot(times, self.angular_velocities, label="Angular velocity [rad/s]")
        self.axes[1].legend(bbox_to_anchor=(0.5, 1.04), loc="lower center")

        # external = self.obsv.get("external")
        # if external is not None and len(external.shape) == 3 and external.dtype == np.uint8:
        #     self.axes[2].imshow(external.transpose([1, 2, 0]))
        # else:
        #     self.axes[2].set_axis_off()

        dt = time() - self.last_render_at
        self.last_render_at = time()
        fps = 1 / dt
        self.fig.suptitle('FPS: {:.2f} '.format(fps))

        self.fig.tight_layout()
        self.fig.canvas.draw()

        plt.pause(0.01)
        # while not plt.waitforbuttonpress(0.01):
        #     pass

        if mode == "human":
            self.fig.canvas.flush_events()
        else:
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data

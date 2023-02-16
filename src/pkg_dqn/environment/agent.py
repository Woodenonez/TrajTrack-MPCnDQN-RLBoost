import numpy as np
from shapely.geometry import Point

from numpy.typing import ArrayLike, NDArray


class MobileRobotSpecification:
    RADIUS = 0.5
    SPEED_MIN = -0.5
    SPEED_MAX = 1.5
    ANGULAR_VELOCITY_MIN = -0.5
    ANGULAR_VELOCITY_MAX = 0.5
    ACCELERATION_MIN = -1.0
    ACCELERATION_MAX = 1.0
    ANGULAR_ACCELERATION_MIN = -3.0
    ANGULAR_ACCELERATION_MAX = 3.0


class MobileRobot:
    def __init__(self, state:NDArray=None, initial_action:NDArray=None):
        """
        Args:
            state: (x, y, theta, v, w)
            initial_action: (acc_v, acc_w)
        """
        if state is not None:
            self.state = state
            self._action = initial_action if (initial_action is not None) else np.array([0,0])
        
        self.cfg = MobileRobotSpecification()

    @property
    def state(self) -> NDArray[np.float32]:
        return self._state
    @state.setter
    def state(self, state: NDArray[np.float32]) -> None:
        self._state = state
        self.point = Point(self.position)

    @property
    def position(self) -> NDArray[np.float32]:
        return self.state[:2]
    @position.setter
    def position(self, position: NDArray[np.float32]) -> None:
        self.state[:2] = position
        self.point = Point(position)

    @property
    def angle(self) -> float:
        return self.state[2]
    @angle.setter
    def angle(self, angle: float) -> None:
        self.state[2] = angle

    @property
    def speed(self) -> float:
        return self.state[3]
    @speed.setter
    def speed(self, speed: float) -> None:
        self.state[3] = speed

    @property
    def angular_velocity(self) -> float:
        return self.state[4]
    @angular_velocity.setter
    def angular_velocity(self, angular_velocity: float) -> None:
        self.state[4] = angular_velocity

    @staticmethod
    def motion_model(sampling_time: float, state: NDArray, action: NDArray) -> NDArray:
        """
        Args:
            state: (x, y, theta, v, w) 
            action: (acc_v, acc_w)
        Returns:
            next_state: Same structure as `state`
        """
        ts = sampling_time
        x, y, theta, v, w = state
        acc_v, acc_w = action
        d_state = np.array([v*np.cos(theta), v*np.sin(theta), 
                            w, acc_v, acc_w]) * ts
        next_state = state + d_state
        return next_state
        
    def step_with_same_velocity(self, time_step: float) -> None:
        self.step(4, time_step)

    def step_with_decay_angular_velocity(self, time_step: float, n_step: int) -> None:
        self.angular_velocity *= 0.9
        self.angle += time_step * self.angular_velocity
        self.position += time_step * self.speed * np.asarray((np.cos(self.angle), np.sin(self.angle)))

    def step(self, action_index: int, time_step: float) -> None:
        """
        Ref:
            Check environment for action space definition.
        Return:
            state: The state vector of the mobile robot.

        Action space
        ------------
            There are 9 combinations of angular (ag) and linear (li) accelerations:
                0: `⇖` ag left  + li fore
                1: `⇑` ag keep  + li fore
                2: `⇗` ag right + li fore
                3: `⇐` ag left  + li keep
                4: `-` ag keep  + li keep
                5: `⇒` ag right + li keep
                6: `⇙` ag left  + li back
                7: `⇓` ag keep  + li back
                8: `⇘` ag right + li back
            E.g. Action 1 is keep the angular velocity and accerlate foreward.
        """

        if action_index // 3 == 0: # 0~2
            self.speed += time_step * self.cfg.ACCELERATION_MAX
        if action_index // 3 == 2:
            self.speed += time_step * self.cfg.ACCELERATION_MIN

        if action_index % 3 == 0:
            self.angular_velocity += time_step * self.cfg.ANGULAR_ACCELERATION_MAX
        if action_index % 3 == 2:
            self.angular_velocity += time_step * self.cfg.ANGULAR_ACCELERATION_MIN

        if self.speed > self.cfg.SPEED_MAX:
           self.speed = self.cfg.SPEED_MAX
        if self.speed < self.cfg.SPEED_MIN:
           self.speed = self.cfg.SPEED_MIN

        if self.angular_velocity > self.cfg.ANGULAR_VELOCITY_MAX:
            self.angular_velocity = self.cfg.ANGULAR_VELOCITY_MAX
        if self.angular_velocity < self.cfg.ANGULAR_VELOCITY_MIN:
            self.angular_velocity = self.cfg.ANGULAR_VELOCITY_MIN
        
        self.angle += time_step * self.angular_velocity
        self.position += time_step * self.speed * np.asarray((np.cos(self.angle), np.sin(self.angle)))

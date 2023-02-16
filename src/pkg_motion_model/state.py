from ._state_component import Angle, Speed, AngularSpeed, Acceleration, AngularAcceleration 
from ._state_component import Velocity, Position

class ActionSpeed:
    def __init__(self, speed: Speed, angular_speed: AngularSpeed):
        self.speed = speed
        self.angular_speed = angular_speed


class ActionVelocity:
    def __init__(self, velocity: Velocity):
        self.velocity = velocity
        self.vx = velocity.x
        self.vy = velocity.y


class ActionAcceleration:
    def __init__(self, acceleration: Acceleration, angular_acceleration: AngularAcceleration):
        self.acceleration = acceleration
        self.angular_acceleration = angular_acceleration


class StatePose:
    def __init__(self, position: Position, angle: Angle):
        self.position = position
        self.x = position.x
        self.y = position.y
        self.angle = angle

    def __str__(self):
        return f"({self.position}, {self.angle})"

    def __eq__(self, other: "StatePose"):
        return self.position == other.position and self.angle == other.angle

    def __call__(self, degrees=False):
        return self.x, self.y, self.angle(degrees)

    @classmethod
    def from_xy(cls, x, y, angle):
        return cls(Position(x, y), Angle(angle))

    @classmethod
    def from_tuple(cls, tuple_state: tuple):
        if len(tuple_state) != 3:
            raise ValueError("Pose requires 3 elements (x, y, angle)")
        return cls(Position(*tuple_state[:2]), Angle(2))
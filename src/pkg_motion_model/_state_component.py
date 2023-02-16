from abc import ABC, abstractmethod

import math

from typing import Union


class Scalar(ABC):
    def __init__(self, value: float):
        self.value = value

    def __str__(self):
        return f"{self.__class__.__name__} ({self.value})"

    def __call__(self):
        return self.value

    def __add__(self, other: "Scalar"):
        return Scalar(self.value + other.value)

    def __sub__(self, other: "Scalar"):
        return Scalar(self.value - other.value)

    def __mul__(self, factor: float):
        return Scalar(self.value * factor)

    def __truediv__(self, factor: float):
        return Scalar(self.value / factor)

    def __eq__(self, other: "Scalar"):
        return self.value == other.value


class Vector2D(ABC):
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self):
        return f"{self.__class__.__name__} ({self.x}, {self.y})"

    def __call__(self):
        return self.x, self.y

    def __add__(self, other: "Vector2D"):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D"):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, factor: float):
        return Vector2D(self.x * factor, self.y * factor)

    def __truediv__(self, factor: float):
        return Vector2D(self.x / factor, self.y / factor)

    def __eq__(self, other: "Vector2D"):
        return self.x == other.x and self.y == other.y



class Angle(Scalar):
    def __init__(self, angle, normalize=False):
        super().__init__(angle)
        if normalize:
            self._within_two_pi()

    def __str__(self, degrees=False):
        if degrees:
            return f"{math.degrees(self.value)} deg"
        return f"{self.value} rad"

    def __call__(self, degrees=False):
        if degrees:
            return math.degrees(self.value)
        super().__call__()

    def _within_two_pi(self):
        self.value = self.value % (2 * math.pi)

    @classmethod
    def from_degrees(cls, angle):
        return cls(math.radians(angle))


class Speed(Scalar):
    def __init__(self, speed):
        super().__init__(speed)

    def __str__(self):
        return f"{self.value} m/s"

    @classmethod
    def from_velocity(cls, velocity: "Velocity"):
        return cls(math.hypot(velocity.x, velocity.y))


class AngularSpeed(Scalar):
    def __init__(self, speed):
        super().__init__(speed)

    def __str__(self, degrees=False):
        if degrees:
            return f"{math.degrees(self.value)} deg/s"
        return f"{self.value} rad/s"

    def __call__(self, degrees=False):
        if degrees:
            return math.degrees(self.value)
        super().__call__()


class Acceleration(Scalar):
    def __init__(self, acceleration):
        super().__init__(acceleration)

    def __str__(self):
        return f"{self.value} m/s^2"

    def to_vector(self, angle: Angle) -> tuple:
        return (self.value * math.cos(angle), self.value * math.sin(angle))


class AngularAcceleration(Scalar):
    def __init__(self, acceleration):
        super().__init__(acceleration)

    def __str__(self, degrees=False):
        if degrees:
            return f"{math.degrees(self.value)} deg/s^2"
        return f"{self.value} rad/s^2"

    def __call__(self, degrees=False):
        if degrees:
            return math.degrees(self.value)
        super().__call__()



class Velocity(Vector2D):
    def __init__(self, x, y):
        super().__init__(x, y)

    @classmethod
    def from_speed_angle(cls, speed: Speed, angle: Angle):
        return cls(speed * math.cos(angle), speed * math.sin(angle))


class Position(Vector2D):
    def __init__(self, x, y):
        super().__init__(x, y)

    def __add__(self, other: "Velocity"):
        if isinstance(other, Velocity):
            super().__add__(other)
        else:
            raise TypeError("Can only add a Velocity to a Position")

    def __sub__(self, other: Union["Position", "Velocity"]) -> float:
        if isinstance(other, Position):
            return math.hypot(self.x - other.x, self.y - other.y)
        elif isinstance(other, Velocity):
            super().__sub__(other)
        else:
            raise TypeError("Can only subtract a Position or a Velocity from a Position")
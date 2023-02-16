import matplotlib.pyplot as plt
import matplotlib.patches as patches

from shapely.geometry import Point, Polygon

from .geometry_plain import PlainPoint, PlainPolygon, PlainCircle, PlainEllipse

from typing import List, Tuple, Union
from matplotlib.axes import Axes


class ShapelyPoint(Point):
    def __init__(self, *args) -> None:
        Point.__init__(self, *args)

    def __call__(self) -> tuple:
        """Return the coordinates of the point."""
        return self.x, self.y

    @classmethod
    def from_plain_point(cls, plain_point: PlainPoint) -> 'ShapelyPoint':
        """Create a ShapelyPoint from a PlainPoint."""
        return cls(plain_point.x, plain_point.y)

    def plot(self, ax: Axes, **kwargs) -> None:
        """Plot the point on the given axes."""
        ax.plot(self.x, self.y, **kwargs)


class ShapelyPolygon(Polygon):
    def __init__(self, vertices, angle:float=0) -> None:
        Polygon.__init__(self, shell=vertices, holes=None)
        self._angle = angle

    @property
    def angle(self) -> float:
        return self._angle

    def __call__(self) -> List[tuple]:
        """Return the coordinates of the polygon."""
        return list(self.exterior.coords[:-1])

    @classmethod
    def from_plain_polygon(cls, plain_polygon: PlainPolygon) -> 'ShapelyPolygon':
        """Create a ShapelyPolygon from a PlainPolygon."""
        return cls(plain_polygon())

    @classmethod
    def from_shapely(cls, polygon: Polygon, angle:float=0) -> 'ShapelyPolygon':
        """Create a ShapelyPolygon from a shapely Polygon."""
        return cls(list(polygon.exterior.coords[:-1]), angle=angle)

    def to_shapely(self) -> Polygon:
        """Create a shapely Polygon from a ShapelyPolygon."""
        return Polygon(list(self.exterior.coords[:-1]))

    def plot(self, ax: Axes, **kwargs) -> None:
        """Plot the polygon on the given axes."""
        ax.plot(*self.exterior.xy, **kwargs)


class ShapelyEllipse(Polygon):
    """A polygonal approximation (based on shapely) of an ellipse."""
    def __init__(self, center: tuple, radii: tuple, angle: float, n:int=16) -> None:
        self._center = center
        self._radii = radii
        self._angle = angle
        self._n_approx = n
        vertices = self._get_approximation(n)
        Polygon.__init__(self, shell=vertices, holes=None)

    @property
    def center(self) -> tuple:
        return self._center

    @property
    def radii(self) -> tuple:
        return self._radii

    @property
    def angle(self) -> float:
        return self._angle

    def __call__(self) -> Tuple[tuple, tuple, float]:
        """Return the center, radii and angle of the ellipse."""
        return self._center, self._radii, self._angle

    def _get_approximation(self, n: int) -> List[tuple]:
        """Return a polygonal approximation of the ellipse."""
        plain_polygon_approximation = PlainEllipse(PlainPoint(*self._center), self._radii, self._angle).return_polygon_approximation(n)
        return plain_polygon_approximation()

    @classmethod
    def from_plain_ellipse(cls, plain_ellipse: PlainEllipse, n:int=16) -> 'ShapelyEllipse':
        """Create a ShapelyEllipse from a PlainEllipse."""
        return cls(plain_ellipse.center, plain_ellipse.radii, plain_ellipse.angle, n=n)

    def to_shapely(self) -> Polygon:
        """Create a shapely Polygon from a ShapelyEllipse."""
        return Polygon(self._get_approximation(self._n_approx))

    def plot(self, ax: Axes, approx:bool=False, **kwargs) -> None:
        """Plot the ellipse on the given axes."""
        if approx:
            ax.plot(*self.exterior.xy, **kwargs)
        else:
            ax.add_patch(patches.Ellipse(self._center, self._radii[0]*2, self._radii[1]*2, self._angle, **kwargs))

    
class ShapelyCircle(ShapelyEllipse):
    """A polygonal approximation (based on shapely) of a circle."""
    def __init__(self, center: tuple, radius: float, angle:float=0, n:int=16) -> None:
        self._radius = radius
        ShapelyEllipse.__init__(self, center, (radius, radius), angle, n=n)

    @property
    def radius(self) -> float:
        return self._radius

    def __call__(self) -> Tuple[tuple, float]:
        """Return the center, radius, and angle of the circle."""
        return self._center, self._radius, self._angle

    @classmethod
    def from_plain_circle(cls, plain_circle: PlainCircle, angle:float=0, n:int=16) -> 'ShapelyCircle':
        """Create a ShapelyCircle from a PlainCircle."""
        return cls(plain_circle.center, plain_circle.radius, angle, n=n)


 

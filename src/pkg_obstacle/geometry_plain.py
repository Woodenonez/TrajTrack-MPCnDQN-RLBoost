from abc import ABC, abstractmethod

import math
from typing import List, Tuple, Union


class PlainGeometry(ABC):
    """A plain geometry class without any dependencies. It is the base class for all other geometry classes."""
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> str:
        return f'{self.__class__.__name__} object'

    @abstractmethod
    def __call__(self) -> None:
        return None

    def return_polygon_approximation(self, n:int=10) -> 'PlainPolygon':
        """Return a polygon approximation of the geometry."""
        pass

    def contains_point(self, point:'PlainPoint') -> bool:
        """Check if the geometry contains the given point."""
        pass


class PlainPoint(PlainGeometry):
    """A plain point class without any dependencies.

    Supported magic methods:
        `str`, `call`, `getitem`, `eq`, `sub`
    """
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f'{self.__class__.__name__} object ({self.x},{self.y})'

    def __call__(self) -> tuple:
        return (self.x, self.y)

    def __getitem__(self, idx) -> float:
        return (self.x, self.y)[idx]

    def __eq__(self, other_point:'PlainPoint') -> bool:
        return self.x == other_point.x and self.y == other_point.y

    def __sub__(self, other_point:'PlainPoint') -> float:
        return math.hypot(self.x-other_point.x, self.y-other_point.y)


class PlainPolygon(PlainGeometry):
    """A plain polygon class without any dependencies.
    
    Supported magic methods:
        `str`, `call`, `getitem`, `eq`
    """
    def __init__(self, vertices: List[PlainPoint]) -> None:
        self.vertices = vertices
        self.ngon = len(self.vertices) # the number of vertices

    def __str__(self) -> str:
        return f'{self.__class__.__name__} object ({[x() for x in self.vertices]})'

    def __call__(self) -> List[tuple]:
        return [x() for x in self.vertices]

    def __getitem__(self, idx) -> PlainPoint:
        return self.vertices[idx]

    def __eq__(self, other_polygon:'PlainPolygon') -> bool:
        return self.vertices == other_polygon.vertices


class PlainEllipse(PlainGeometry):
    """A plain ellipse class without any dependencies.

    Supported magic methods:
        `str`, `call`, `eq`

    Methods:
        `return_polygon_approximation`: return a (inscribed) polygon approximation of the ellipse
        'contains_point': check if the ellipse contains the point
    """
    def __init__(self, center: PlainPoint, radii: tuple, angle: float) -> None:
        self.center = center
        self.radii = radii
        self.angle = angle

    def __str__(self) -> str:
        return f'{self.__class__.__name__} object (c={self.center()}, r={self.radii}, a={self.angle})'

    def __call__(self) -> Tuple[tuple, tuple, float]:
        return (self.center(), self.radii, self.angle)
    
    def __eq__(self, other_ellipse:'PlainEllipse') -> bool:
        return self.center == other_ellipse.center and self.radii == other_ellipse.radii and self.angle == other_ellipse.angle

    def return_polygon_approximation(self, n:int=10) -> PlainPolygon:
        """Return a polygon approximation of the ellipse."""
        x, y = self.center()
        rx, ry = self.radii
        a = self.angle
        ellipse_samples = [(rx*math.cos(2*math.pi*i/n), ry*math.sin(2*math.pi*i/n)) for i in range(n)]
        rotation_matrix = [[math.cos(a), -math.sin(a)], 
                           [math.sin(a), math.cos(a)]]
        ellipse_samples = [PlainPoint(x + rotation_matrix[0][0]*sample[0] + rotation_matrix[0][1]*sample[1], 
                                      y + rotation_matrix[1][0]*sample[0] + rotation_matrix[1][1]*sample[1]) for sample in ellipse_samples]
        return PlainPolygon(ellipse_samples)

    def contains_point(self, point: PlainPoint, value:bool=False) -> Union[bool, float]:
        """Check if the ellipse contains the point. 
        
        If `value` is True, return a value. Positive value means the point is inside the ellipse."""
        x, y = self.center()
        rx, ry = self.radii
        a = self.angle
        rotation_matrix = [[math.cos(a), -math.sin(a)], 
                           [math.sin(a), math.cos(a)]]
        pt = PlainPoint(rotation_matrix[0][0]*(point.x-x) + rotation_matrix[0][1]*(point.y-y), 
                        rotation_matrix[1][0]*(point.x-x) + rotation_matrix[1][1]*(point.y-y))
        if value:
            return 1 - (pt.x/rx)**2 - (pt.y/ry)**2
        return (pt.x/rx)**2 + (pt.y/ry)**2 <= 1


class PlainCircle(PlainGeometry):
    """A plain circle class without any dependencies.
    
    Supported magic methods:
        `str`, `call`, `eq`

    Methods:
        `return_polygon_approximation`: return a (inscribed-default/circumscribed) polygon approximation of the circle
        `contains_point`: check if the circle contains the point
    """
    def __init__(self, center: PlainPoint, radius:float) -> None:
        self.center = center
        self.radius = radius

    def __str__(self) -> str:
        return f'{self.__class__.__name__} object (c={self.center()}, r={self.radius})'

    def __call__(self) -> Tuple[tuple, float]:
        return (self.center(), self.radius)

    def __eq__(self, other_circle:'PlainCircle') -> bool:
        return self.center == other_circle.center and self.radius == other_circle.radius

    def return_polygon_approximation(self, n:int=10, inscribed:bool=True) -> PlainPolygon:
        """Return a polygon approximation of the circle. If not inscribed, it is circumscribed."""
        if inscribed:
            return self._return_inscribed_regular_polygon(n)
        else:    
            return self._return_circumscribed_regular_polygon(n)

    def _return_inscribed_regular_polygon(self, n:int=10) -> PlainPolygon:
        vertices = [PlainPoint(self.center.x + self.radius*math.cos(2*math.pi/n*i), self.center.y + self.radius*math.sin(2*math.pi/n*i)) for i in range(n)]
        return PlainPolygon(vertices)

    def _return_circumscribed_regular_polygon(self, n:int=10) -> PlainPolygon:
        """It can be seen as an inscribed polygon of a larger circle.
        """
        vertices = [PlainPoint(self.center.x + (self.radius/math.cos(math.pi/n))*math.cos(2*math.pi/n*i), self.center.y + (self.radius/math.cos(math.pi/n))*math.sin(2*math.pi/n*i)) for i in range(n)]
        return PlainPolygon(vertices)

    def contains_point(self, point: PlainPoint, value:bool=False) -> Union[bool, float]:
        """Check if the circle contains the point. 

        If `value` is True, 
        return the difference of (i) the distance from the point to the certer and (ii) the radius.
        """
        if value:
            return math.hypot(self.center.x-point.x, self.center.y-point.y) - self.radius
        return math.hypot(self.center.x-point.x, self.center.y-point.y) <= self.radius



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    list_of_points = [PlainPoint(1,2), PlainPoint(1,1), PlainPoint(1,0), PlainPoint(0,1)]
    polygon = PlainPolygon(list_of_points)

    circle = PlainCircle(PlainPoint(0,0), 1)
    circle_approx = circle.return_polygon_approximation(100)
    polygon_inscribed = circle.return_polygon_approximation(6, inscribed=True)
    polygon_circumscribed = circle.return_polygon_approximation(6, inscribed=False)

    ellipse = PlainEllipse(PlainPoint(-1,-2), (1,2), math.pi/4)
    ellipse_approx = ellipse.return_polygon_approximation(100)

    print(ellipse)

    plt.figure()
    plt.plot(*zip(*polygon()), 'o-')

    plt.plot(*zip(*circle_approx()), '-')
    plt.plot(*zip(*polygon_inscribed()), 'x-')
    plt.plot(*zip(*polygon_circumscribed()), 'o-')

    plt.plot(*zip(*ellipse_approx()), '-')

    plt.axis('equal')
    plt.show()
import math

import shapely.affinity
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon
from shapely.geometry import JOIN_STYLE

from typing import List, Tuple, Union


def polygon_to_vertices(polygon: Polygon) -> List[tuple]:
    return list(polygon.exterior.coords)[:-1]

def point_inflate(point: Point, inflation_margin: float, n: int=4) -> Point:
    """Inflate a point (to a regular polygon) by a given positive margin.
    
    The number of vertices of the polygon is given by 4*n.
    """
    return point.buffer(inflation_margin, resolution=n)

def polygon_inflate(polygon: Polygon, inflation_margin: float, method:JOIN_STYLE=JOIN_STYLE.mitre) -> Polygon:
    """Inflate a polygon by a given margin. If the margin is negative, the polygon will be deflated."""
    return polygon.buffer(inflation_margin, join_style=method)

def geometry_translate(geometry: BaseGeometry, translation: tuple) -> BaseGeometry:
    """Translate a shapely geometry by a given translation vector."""
    return shapely.affinity.translate(geometry, translation[0], translation[1])

def geometry_rotate(geometry: BaseGeometry, angle: float, origin:tuple=(0,0)) -> BaseGeometry:
    """Rotate a shapely geometry by a given angle around a given origin."""
    return shapely.affinity.rotate(geometry, angle, origin=origin, use_radians=True)

def geometry_frame_transform(geometry: BaseGeometry, origin_A: tuple, origin_B: tuple, angle_A: float, angle_B: float) -> BaseGeometry:
    """Transform a shapely geometry from a given frame A to a given frame B.
    Frames A and B are defined by the relative origins and the angles to the standard world frame.

    This is equivalent to:
        First translate along vector(BA) and rotate -angle(B-A) to frame A.
    """
    rotation_angle = angle_A - angle_B
    translation_vector = (origin_A[0] - origin_B[0], origin_A[1] - origin_B[1])
    return geometry_rotate(geometry_translate(geometry, translation_vector), rotation_angle)

def geometry_affine_transform(geometry: BaseGeometry, angle: float, xoff: float, yoff: float) -> BaseGeometry:
    """Rotate a shapely geometry by a given angle around a given origin.

    This is equivalent to (1) or (2):
        (1): translate a rotated geometry by a given translation vector.
        (2): rotate a translated geometry by a given angle around the translation vector point.
    """
    transform_mtx_param = [math.cos(angle), -math.sin(angle), math.sin(angle), math.cos(angle), xoff, yoff]
    return shapely.affinity.affine_transform(geometry, transform_mtx_param)


if __name__ == "__main__":
    import math
    import matplotlib.pyplot as plt

    pt = Point(-1, 0)
    pt_inflated = point_inflate(pt, 0.5, 4)

    poly = Polygon([(1, 1), (1, 2), (2, 2), (2, 1)])
    poly_inflated = polygon_inflate(poly, 0.5)
    poly_traslated = geometry_translate(poly, (1, 1))
    poly_rotated = geometry_rotate(poly, math.pi/4)

    poly_affine1 = geometry_rotate(poly_traslated, math.pi/4, origin=(1, 1))
    poly_affine2 = geometry_translate(poly_rotated, (1, 1))
    poly_affine = geometry_affine_transform(poly, math.pi/4, 1, 1)

    poly_transform_frame = geometry_frame_transform(poly, (0, 0), (1, 1), 0, math.pi/4)

    plt.figure()
    plt.plot(*zip(*polygon_to_vertices(pt_inflated)), 'x-')
    plt.plot(*zip(*polygon_to_vertices(poly)), 'x-')
    # plt.plot(*zip(*polygon_to_vertices(poly_inflated)), 'o-')
    plt.plot(*zip(*polygon_to_vertices(poly_traslated)), 'ro-')
    plt.plot(*zip(*polygon_to_vertices(poly_rotated)), 'go-')
    plt.plot(*zip(*polygon_to_vertices(poly_affine)), 'x-')
    plt.plot(*zip(*polygon_to_vertices(poly_affine1)), '*-')
    plt.plot(*zip(*polygon_to_vertices(poly_affine2)), '+-')
    plt.plot(*zip(*polygon_to_vertices(poly_transform_frame)), '--')
    plt.axis('equal')
    plt.show()

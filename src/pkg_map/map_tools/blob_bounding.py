import numpy as np
from scipy import spatial
from skimage import util, color, filters, measure, morphology

from typing import List, Tuple, Union


class BlobBounding:
    def __init__(self, bounding_degree=4) -> None:
        """
        Args:
            bounding_degree: The number of vertices that a bounding polygon should have.
        """
        self.n_vertices = bounding_degree

    @staticmethod
    def get_edge_map(binary_image: np.ndarray, dilation_size:int=3):
        """Get the edge map of a binary image.
        Args:
            binary_image: An image with 0s and 1s. Edge detection is applied on pixels of 1s.
            dilation_size: Dilate the image before finding the edge.
        Returns:
            edge_map: An image of the same size as the input, but all 1s are at the edges.
        """
        if dilation_size > 0:
            edge_map = filters.roberts(morphology.dilation(binary_image, np.ones((dilation_size, dilation_size))))
        else:
            edge_map = filters.roberts(binary_image)
        return util.invert(edge_map)

    @staticmethod
    def get_bounding_rectangle(hull_points: np.ndarray):
        """Find the smallest bounding rectangle for a convex hull.
        Ref:
            https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
        Args:
            hull_points: an n*2 matrix of coordinates.
        Returns:
            rval: An n*2 matrix of coordinates of rectangle vertices.
        """
        # calculate edge angles
        edges = np.zeros((len(hull_points)-1, 2))
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.zeros((len(edges)))
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, np.pi/2))
        angles = np.unique(angles)

        # find rotation matrices
        rotations = np.vstack([
            np.cos(angles),
            np.cos(angles-np.pi/2),
            np.cos(angles+np.pi/2),
            np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))

        # apply rotations to the hull
        rot_points = np.dot(rotations, hull_points.T)

        # find the bounding points
        min_x = np.nanmin(rot_points[:, 0], axis=1)
        max_x = np.nanmax(rot_points[:, 0], axis=1)
        min_y = np.nanmin(rot_points[:, 1], axis=1)
        max_y = np.nanmax(rot_points[:, 1], axis=1)

        # find the box with the best area
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)

        # return the best box
        x1 = max_x[best_idx]
        x2 = min_x[best_idx]
        y1 = max_y[best_idx]
        y2 = min_y[best_idx]
        r = rotations[best_idx]

        rval = np.zeros((4, 2))
        rval[0] = np.dot([x1, y2], r)
        rval[1] = np.dot([x2, y2], r)
        rval[2] = np.dot([x2, y1], r)
        rval[3] = np.dot([x1, y1], r)

        return rval

    def get_bounding_polygon(self, hull_points: np.ndarray):
        # coords = measure.approximate_polygon(contour, tolerance=5)
        if self.n_vertices == 4:
            return self.get_bounding_rectangle(hull_points)
        else:
            raise NotImplementedError("Only support rectangle bouning box now.")

    def get_bounding_polygons(self, grayscale_image: np.ndarray) -> List[np.ndarray]:
        polygons = []
        for contour in measure.find_contours(grayscale_image):
            hull_points = contour[spatial.ConvexHull(contour).vertices, :]
            coords = self.get_bounding_polygon(hull_points)
            polygons.append(coords[:, ::-1]) # contour need x-y swap
        return polygons
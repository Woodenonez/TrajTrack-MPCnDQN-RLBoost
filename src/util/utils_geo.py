from typing import List, Tuple, Union

import numpy as np
from scipy import spatial

def lineseg_dists(points:np.ndarray, line_points_1:np.ndarray, line_points_2:np.ndarray) -> np.ndarray:
    '''Cartesian distance from point to line segment \\
    Edited to support arguments as series, from: https://stackoverflow.com/a/54442561/11208892 \\
    Argument
        p: np.array of shape (n_p, 2)
        a: np.array of shape (n_l, 2)
        b: np.array of shape (n_l, 2)
    Return
        o: np.array of shape (n_p, n_l)
    '''
    p, a, b = points, line_points_1, line_points_2
    if len(p.shape) < 2:
        p = p.reshape(1,2)
    n_p, n_l = p.shape[0], a.shape[0]
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
    # signed parallel distance components, rowwise dot products of 2D vectors
    s = np.multiply(np.tile(a, (n_p,1)) - p.repeat(n_l, axis=0), np.tile(d, (n_p,1))).sum(axis=1)
    t = np.multiply(p.repeat(n_l, axis=0) - np.tile(b, (n_p,1)), np.tile(d, (n_p,1))).sum(axis=1)
    # clamped parallel distance
    h = np.amax([s, t, np.zeros(s.shape[0])], axis=0)
    # perpendicular distance component, rowwise cross products of 2D vectors  
    d_pa = p.repeat(n_l, axis=0) - np.tile(a, (n_p,1))
    c = d_pa[:, 0] * np.tile(d, (n_p,1))[:, 1] - d_pa[:, 1] * np.tile(d, (n_p,1))[:, 0]
    return np.hypot(h, c).reshape(n_p, n_l)

def polygon_halfspace_representation(polygon_points:np.ndarray):
    '''Compute the H-representation of a set of points (facet enumeration). \\
    Reference
        Code: https://github.com/d-ming/AR-tools/blob/master/artools/artools.py
    Return
        A: (L x d) array. Each row in A represents hyperplane normal.
        b: (L x 1) array. Each element in b represents the hyperpalne constant bi
    '''
    hull = spatial.ConvexHull(polygon_points)
    hull_center = np.mean(polygon_points[hull.vertices, :], axis=0)  # (1xd) vector
    
    K = hull.simplices
    V = polygon_points - hull_center # perform affine transformation
    A = np.nan * np.empty((K.shape[0], polygon_points.shape[1]))

    rc = 0
    for i in range(K.shape[0]):
        ks = K[i, :]
        F = V[ks, :]
        if np.linalg.matrix_rank(F) == F.shape[0]:
            f = np.ones(F.shape[0])
            A[rc, :] = np.linalg.solve(F, f)
            rc += 1

    A:np.ndarray = A[:rc, :]
    b:np.ndarray = np.dot(A, hull_center.T) + 1.0
    return b.tolist(), A[:,0].tolist(), A[:,1].tolist()

def decompose_convex_polygons(original_vertices:np.ndarray, num_vertices_new:int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    '''
    Return
        poly_list    : a list of decomposed polygons, each polygon is a array of vertices
        poly_list_vis: a list of decomposed polygons (append with the first vertice) for visualization
    '''
    if num_vertices_new < 3:
        raise ValueError(f'The number of edges of a polygon must be larger than 2, got {num_vertices_new}')
    if num_vertices_new >= original_vertices.shape[0]:
        return [original_vertices], [np.concatenate((original_vertices, original_vertices[[0], :]), axis=0)]
    closed_vertices = np.concatenate((original_vertices, original_vertices[[0,1], :]), axis=0)
    n_real = closed_vertices.shape[0]
    n_new = num_vertices_new 
    current_idx = 0
    poly_list = []
    while current_idx>=0:
        if (current_idx+n_new) > n_real:
            poly:np.ndarray = closed_vertices[current_idx:, :]
            if poly.shape[0] < 2:
                poly = np.concatenate((poly, closed_vertices[:2, :]), axis=0)
            elif poly.shape[0] < 3:
                poly = np.concatenate((poly, closed_vertices[[0], :]), axis=0)
            current_idx = -1
        else:
            poly = closed_vertices[current_idx:(current_idx+n_new), :]
            current_idx += n_new-1-1 # more robust
        poly_list.append(poly)
    poly_list_vis = [np.concatenate((poly, poly[[0],:]), axis=0) for poly in poly_list]
    return poly_list, poly_list_vis

def minimum_bounding_rectangle(hull_points:np.ndarray) -> np.ndarray:
    '''Find the smallest bounding rectangle for a convex hull. \\
    Reference
        Code: [Algorithm to find the minimum-area-rectangle for given points in order to compute the major and minor axis length] in Stackoverflow
    Argument
        hull_points: an n*2 matrix of coordinates
    Return
        rval: an n*2 matrix of coordinates
    '''
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


class CoordTransform:
    def __init__(self, scale:float=1, offsetx_after:float=0, offsety_after:float=0, x_reverse=False, y_reverse=False, x_max_before=0, y_max_before=0):
        '''Transform the given coordinates by some scales and offsets.
        Argument
            scale       : Scale factor.
            offset_after: The offset of x and y axes.
            reverse     : If the x and y axes should be reversed.
            max_before  : The maximal values along x and y axes, used to calculate the reversed coordinates.
        Comment
            For orginal coordinates z=[x,y], if x or y is reversed, calculate the reversed coordinate first.
            Then calculate the transformed coordinates according to the scaling and the offset.
        '''
        self.k = [scale, scale]
        self.b = [offsetx_after, offsety_after]
        self.xr = x_reverse
        self.yr = y_reverse
        self.xm = x_max_before
        self.ym = y_max_before

    def __call__(self, state:Union[list, np.ndarray], forward=True) -> Union[list, np.ndarray]:
        '''Return the transformed state. If forward=False, it means from the transformed state to the original one.
        '''
        tr_state = state.copy()
        if forward:
            if self.xr:
                tr_state[0] = self.xm - tr_state[0]
            if self.yr:
                tr_state[1] = self.ym - tr_state[1]
            tr_state[0] = tr_state[0]*self.k[0]+self.b[0]
            tr_state[1] = tr_state[1]*self.k[1]+self.b[1]
        else:
            tr_state[0] = (state[0]-self.b[0]) / self.k[0]
            tr_state[1] = (state[1]-self.b[1]) / self.k[1]
            if self.xr:
                tr_state[0] = self.xm - tr_state[0]
            if self.yr:
                tr_state[1] = self.ym - tr_state[1]
        return tr_state

    def cvt_coord_x(self, x:np.ndarray, forward=True) -> np.ndarray:
        if forward:
            if self.xr:
                x = self.xm - x
            cvt_x = self.k[0]*x + self.b[0]
        else:
            cvt_x = (x-self.b[0]) / self.k[0]
            if self.xr:
                cvt_x = self.xm - cvt_x
        return cvt_x

    def cvt_coord_y(self, y:np.ndarray, forward=True) -> np.ndarray:
        if forward:
            if self.yr:
                y = self.ym - y
            cvt_y = self.k[1]*y + self.b[1]
        else:
            cvt_y = (y-self.b[1]) / self.k[1]
            if self.yr:
                cvt_y = self.ym - cvt_y
        return cvt_y

    def cvt_coords(self, x:np.ndarray, y:np.ndarray, forward=True) -> np.ndarray:
        '''Return transformed/original coordinates, in shape (2*n).
        '''
        cvt_x = self.cvt_coord_x(x, forward)
        cvt_y = self.cvt_coord_y(y, forward)
        return np.hstack((cvt_x[:,np.newaxis], cvt_y[:,np.newaxis]))

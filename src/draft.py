import sys
import math
import random
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageFilter
import casadi as cs
from scipy import interpolate

from shapely.geometry import LinearRing, Point, Polygon, LineString

print([(v, w) for (v, w) in zip(range(10), range(1,11))])

sys.exit(0)

def move_x0_outside_of_closest_border(x0, OneObstacle):
    # print('x0 is inside obstacle, old x0' , x0)
    # Point on the border:
    pol_ext = LinearRing(OneObstacle.exterior.coords)
    d = pol_ext.project(Point(x0[0], x0[1]))
    p = pol_ext.interpolate(d)
    closest_point_coords = list(p.coords)[0]

    x0[0] = x0[0] + 2 * (closest_point_coords[0] - x0[0])
    x0[1] = x0[1] + 2 * (closest_point_coords[1] - x0[1])

    return x0

pt = [0.1, 0.9]
polygon = Polygon([[0,0], [0,1], [1,1]])
pt1 = move_x0_outside_of_closest_border(deepcopy(pt), polygon)

plt.plot(polygon.exterior.coords.xy[0], polygon.exterior.coords.xy[1])
plt.plot(pt[0], pt[1], 'rx')
plt.plot(pt1[0], pt1[1], 'gx')
plt.axis('equal')
plt.show()

sys.exit(0)

fig = plt.figure()
ax = fig.add_subplot()
ax.autoscale() 

hl, = ax.plot([],[], '*')
vl, = ax.plot([],[], 'rx')

vl.set_xdata([0.01])
vl.set_ydata([0.01])

for i in range(10):
    hl.set_xdata(np.append(hl.get_xdata(),  i*0.01))
    hl.set_ydata(np.append(hl.get_ydata(), -i*0.01))
    plt.draw()
    plt.pause(0.2)

hl.set_xdata([])
hl.set_ydata([])
plt.draw()

plt.show()

sys.exit(0)

### Generate random convex polygons
#Ref: Probability that n random points are in convex position (1994)
def generateConvex(n: int):
    '''
    Generate convex shappes according to Pavel Valtr's 1995 alogrithm. Ported from
    Sander Verdonschot's Java version, found here:
    https://cglab.ca/~sander/misc/ConvexGeneration/ValtrAlgorithm.java
    '''

    random.seed(0)
    # initialise random coordinates
    X_rand, Y_rand = np.sort(np.random.random(n)), np.sort(np.random.random(n))
    X_new, Y_new = np.zeros(n), np.zeros(n)

    # divide the interior points into two chains
    last_true = last_false = 0
    for i in range(1, n):
        if i != n - 1:
            if random.getrandbits(1):
                X_new[i] = X_rand[i] - X_rand[last_true]
                Y_new[i] = Y_rand[i] - Y_rand[last_true]
                last_true = i
            else:
                X_new[i] = X_rand[last_false] - X_rand[i]
                Y_new[i] = Y_rand[last_false] - Y_rand[i]
                last_false = i
        else:
            X_new[0] = X_rand[i] - X_rand[last_true]
            Y_new[0] = Y_rand[i] - Y_rand[last_true]
            X_new[i] = X_rand[last_false] - X_rand[i]
            Y_new[i] = Y_rand[last_false] - Y_rand[i]

    # randomly combine x and y and sort by polar angle
    np.random.shuffle(Y_new)
    vertices = np.stack((X_new, Y_new), axis=-1)
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]

    # arrange points end to end to form a polygon
    vertices = np.cumsum(vertices, axis=0)

    # center around the origin
    x_max, y_max = np.max(vertices[:, 0]), np.max(vertices[:, 1])
    vertices[:, 0] += ((x_max - np.min(vertices[:, 0])) / 2) - x_max
    vertices[:, 1] += ((y_max - np.min(vertices[:, 1])) / 2) - y_max

    return vertices

def decompose_convex_polygons(original_vertices:np.ndarray, num_vertices_new:int):
    if num_vertices_new < 3:
        raise ValueError(f'The number of edges of a polygon must be larger than 2, got {num_vertices_new}')
    if num_vertices_new >= original_vertices.shape[0]:
        return [original_vertices], [np.concatenate((original_vertices, original_vertices[[0], :]), axis=0)]
    close_vertices = np.concatenate((original_vertices, original_vertices[[0,1], :]), axis=0)
    n_real = close_vertices.shape[0]
    n_new = num_vertices_new 
    current_idx = 0
    poly_list = []
    while current_idx>=0:
        if (current_idx+n_new) > n_real:
            poly = close_vertices[current_idx:, :]
            if poly.shape[0] < 2:
                poly = np.concatenate((poly, close_vertices[:2, :]), axis=0)
            elif poly.shape[0] < 3:
                poly = np.concatenate((poly, close_vertices[[0], :]), axis=0)
            current_idx = -1
        else:
            poly = close_vertices[current_idx:(current_idx+n_new), :]
            current_idx += n_new-1-1
        poly_list.append(poly)
    poly_list_vis = [np.concatenate((poly, poly[[0],:]), axis=0) for poly in poly_list]
    return poly_list, poly_list_vis

vertices = generateConvex(10)
vertices_vis = np.concatenate((vertices, vertices[[0],:]), axis=0)
_, poly_list_vis = decompose_convex_polygons(vertices, num_vertices_new=10)

plt.plot(vertices_vis[:,0], vertices_vis[:,1])
for poly_vis in poly_list_vis:
    plt.plot(poly_vis[:,0], poly_vis[:,1], 'r')
    plt.pause(1)

plt.show()

# n_vertices = 6
# view_box = [0, 0, 10, 10] # [xmin, ymin, xmax, ymax]

# print([random.random()*360 for _ in range(n_vertices)])

# def generate_random_convex_polygon(n_vertices:int, view_box:list):
#     xc = random.random() * (view_box[2]-view_box[0]) + view_box[0]
#     yc = random.random() * (view_box[3]-view_box[1]) + view_box[1]
#     random_angles = sorted([random.random()*360 for _ in range(n_vertices)])

# def generate_random_convex_polygon(n_vertices:int, view_box:list):
#     Xs = sorted(random.sample(range(view_box[0], view_box[2]), n_vertices))
#     Ys = sorted(random.sample(range(view_box[1], view_box[3]), n_vertices))

#     Xmin, Xmax = Xs[0], Xs[-1]
#     Ymin, Ymax = Ys[0], Ys[-1]
#     Xothers = random.sample(Xs[1:-1], n_vertices-2)
#     Yothers = random.sample(Ys[1:-1], n_vertices-2)

#     X1 = [Xmin] + sorted(Xothers[:(n_vertices-2)//2]) + [Xmax]
#     X2 = [Xmin] + sorted(Xothers[(n_vertices-2)//2:]) + [Xmax]
#     Y1 = [Ymin] + sorted(Yothers[:(n_vertices-2)//2]) + [Ymax]
#     Y2 = [Ymin] + sorted(Yothers[(n_vertices-2)//2:]) + [Ymax]

#     XVec = [j-i for i, j in zip(X1[:-1], X1[1:])] + [j-i for i, j in zip(X2[1:], X2[:-1])]
#     YVec = [j-i for i, j in zip(Y1[:-1], Y1[1:])] + [j-i for i, j in zip(Y2[1:], Y2[:-1])]

# generate_random_convex_polygon(n_vertices, view_box)

sys.exit(0)

print(random.sample(list(range(1,11)), k=5))
sys.exit(0)

def convolve_1d(signal, kernel):
    kernel = kernel[::-1]
    out = []
    for i in range(1-len(kernel),len(signal)):
        out.append( np.dot(signal[max(0,i):min(i+len(kernel),len(signal))], kernel[max(-i,0):len(signal)-i*(len(signal)-len(kernel)<i)]) )
    return out

base_weight = [200, 100, 50, 20, 10, 8, 5, 4, 3, 2] + [1]*10
kernel = [1, 0.5, 0.5]
for kt in range(20):
    onehot_signal = [0]*20
    onehot_signal[kt] = base_weight[kt]
    horizon_weight = convolve_1d(onehot_signal, kernel=kernel)
    print(horizon_weight)
sys.exit(0)

a = cs.DM([[1,1,1], [2,3,-4]]).T
print(a)
b = cs.mtimes(a, cs.DM([1,1]))
result = 1
for i in range(b.shape[0]):
    result *= cs.fmax(0, b[i])
print(result)
sys.exit(0)


boundary = [(0,0), (0,10), (12,10), (12,0)]
obstacle_list = [[(1,1), (1,3), (3,3), (3,1)],
                 [(5,5), (7,5), (5,7)]]
width  = max(np.array(boundary)[:,0]) - min(np.array(boundary)[:,0])
height = max(np.array(boundary)[:,1]) - min(np.array(boundary)[:,1])

fig, ax = plt.subplots(figsize=(width, height), dpi=100)
ax.set_aspect('equal')
ax.axis('off')
ax.plot(np.array(boundary)[:,0], np.array(boundary)[:,1], 'r-')
for coords in obstacle_list:
    x, y = np.array(coords)[:,0], np.array(coords)[:,1]
    plt.fill(x, y, color='k')

fig.tight_layout(pad=0)

fig.canvas.draw()
image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

print(fig_size, image_from_plot.shape)
plt.close()

sys.exit(0)

def get_closest_edge_point(original_point:np.ndarray, occupancy_map:np.ndarray) -> np.ndarray:
    if len(original_point.shape) == 1:
        original_point = original_point[np.newaxis,:]
    if len(occupancy_map.shape) != 2:
        raise ValueError(f'Input map should be 2-dim (got {len(occupancy_map.shape)}-dim).')
    if original_point.shape[1] != 2:
        raise ValueError(f'Input point shape incorrect (should be 2, got {original_point.shape[1]}).')
    def np_dist_map(centre, base_matrix): # centre's size is nx2, base_matrix is wxh
        npts = centre.shape[0]
        x = np.arange(0, base_matrix.shape[1])
        y = np.arange(0, base_matrix.shape[0])
        x, y = np.meshgrid(x, y)
        x, y = x[:, :, np.newaxis], y[:, :, np.newaxis]
        xc_full_mtx = np.ones_like(x).repeat(npts, axis=2) * np.expand_dims(centre[:,0], axis=(0,1))
        yc_full_mtx = np.ones_like(y).repeat(npts, axis=2) * np.expand_dims(centre[:,1], axis=(0,1))
        base_matrix = (x.repeat(npts, axis=2)-xc_full_mtx)**2 + (y.repeat(npts, axis=2)-yc_full_mtx)**2
        return base_matrix/base_matrix.max()
    occupancy_map /= np.amax(occupancy_map, axis=(0,1), keepdims=True)
    edge_map = np.array(Image.fromarray(occupancy_map.astype('uint8'), 'L')
                        .filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.FIND_EDGES))
    edge_map[edge_map>0] = 1
    dist_map = np_dist_map(original_point, occupancy_map) * edge_map[:,:,np.newaxis].repeat(original_point.shape[0], axis=2)
    dist_map[dist_map==0] = np.max(dist_map)
    closest_edge_point = []
    for i in range(original_point.shape[0]):
        closest_edge_point.append(np.unravel_index(np.argmin(dist_map[:,:,i], axis=None), dist_map[:,:,i].shape))
    return np.array(closest_edge_point)

bad_point = np.array([[65, 40], [60,35]])
map = np.zeros((100,100))
map[30:61, 50:71] = 1
map[20:41, 20:41] = 1
map[35:45, 70:80] = 1

good_point = get_closest_edge_point(bad_point, map)

print(good_point)

_, [ax1, ax2] = plt.subplots(1,2)
[ax.plot(bad_point[:, 0],  bad_point[:, 1],  'rx') for ax in [ax1, ax2]]
[ax.plot(good_point[:, 1], good_point[:, 0], 'go') for ax in [ax1, ax2]]
ax1.imshow(map, cmap='gray')
ax2.imshow(map, cmap='gray')
plt.show()
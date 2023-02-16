import math
import random
import warnings
from typing import Union, List, Dict, Any

import numpy as np
import networkx as nx

# from util.basic_datatype import *

'''
Description:
    All standard commonly-used objects are defined here.
    Each object should be defined strictly with data type hints.
    Each object should have some basic functions (such as __call__, __getitem__, etc.).
    The functions are just designed for Python though!
Comment:
    Python native float is float64 (double float).
    Python list is like C++ array.
'''

def checktype(object:object, desired_type:Union[type, List[type]]) -> object:
    '''Check if the given object has the desired type (if so, return the object).
    '''
    if isinstance(desired_type, type):
        if not isinstance(object, desired_type):
            raise TypeError(f'Input must be a {desired_type}, got {type(object)}.')
    elif isinstance(desired_type, List):
        [checktype(x, type) for x in desired_type]
        [checktype(object, x) for x in desired_type]
    else:
        raise TypeError('Desired type must be a type or a list of types.')
    return object

#%%# Informative objects
class IdentityHeader:
    '''Similar to ROS Header
    '''
    def __init__(self, id:Union[int, str]=None, timestamp:float=None, category:str=None, priority:int=0) -> None:
        self.__prt_name = '[Header]'
        self.__input_validation(id, timestamp, category, priority)
        self.id = id
        self.timestamp = timestamp
        self.category = category
        self.priority = priority
    
    def __input_validation(self, id, timestamp, category, priority):
        if (id is None) and (timestamp is None) and (category is None):
            warnings.warn(f'{self.__prt_name} No information is specified for the object.')
        if (not isinstance(id, (int, str))) and (id is not None):
            raise TypeError(f'ID should be either int or str, got {type(id)}.')
        if (not isinstance(timestamp, float)) and (timestamp is not None):
            raise TypeError(f'Timestamp should be float, got {type(timestamp)}.')
        if (not isinstance(category, str)) and (category is not None):
            raise TypeError(f'Category should be str, got {type(category)}.')
        if (not isinstance(priority, int)):
            raise TypeError(f'Priority should be int, got {type(priority)}.')

#%%# Basic parent classes
class ListLike(list):
    def __init__(self, input_list:list, element_type:Union[type, List[type]]) -> None:
        '''
        Comment
            :A list-like object is a list of elements, where each element can be converted into a tuple.
            :An element should have the __call__ method which returns a tuple.
        '''
        super().__init__(input_list)
        self.elem_type = element_type
        self.__input_validation(input_list, element_type)

    def __input_validation(self, input_list, element_type):
        checktype(input_list, list)
        if input_list:
            [checktype(element, element_type) for element in input_list]

    def __call__(self):
        '''
        Description
            :Convert elements to tuples. Elements must have __call__ method.
        '''
        return [x() for x in self]

    def append(self, element) -> None:
        super().append(checktype(element, self.elem_type))

    def insert(self, position:int, element) -> None:
        super().insert(position, checktype(element, self.elem_type))

    def numpy(self) -> np.ndarray:
        '''
        Description
            :Convert list to np.ndarray. Elements must have __call__ method.
        '''
        return np.array([x() for x in self])


#%%# ROS-style basic geometric objects
class Point2D:
    '''Similar to ROS Point
    '''
    def __init__(self, x:float, y:float) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f'Point2D object ({self.x},{self.y})'

    def __call__(self) -> tuple:
        return (self.x, self.y)

    def __getitem__(self, idx):
        return (self.x, self.y)[idx]

    def __sub__(self, other_point:'Point2D') -> float:
        '''
        Return:
            :The distance between two Point2D objects.
        '''
        return math.hypot(self.x-other_point.x, self.y-other_point.y)

class Pose2D:
    '''Similar to ROS Pose2D
    '''
    def __init__(self, x:float, y:float, theta:float) -> None:
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self) -> str:
        return f'Pose2D object ({self.x},{self.y},{self.theta})'

    def __call__(self) -> tuple:
        return (self.x, self.y, self.theta)

    def __getitem__(self, idx):
        return (self.x, self.y, self.theta)[idx]

    def __sub__(self, other_pose2d:'Pose2D') -> float:
        '''
        Return:
            :The distance between two pose2d objects.
        '''
        return math.hypot(self.x-other_pose2d.x, self.y-other_pose2d.y)

class Pose2DStamped(Pose2D, IdentityHeader):
    '''Stamped objects have the "time" attribute (Similar but not the same as ROS stamped).
    '''
    def __init__(self, x:float, y:float, theta:float, timestamp:float) -> None:
        Pose2D.__init__(self, x, y, theta)
        IdentityHeader.__init__(self, timestamp=timestamp)

class Pose2DMarked(Pose2D, IdentityHeader):
    '''Marked objects have the "id" attribute.
    '''
    def __init__(self, x:float, y:float, theta:float, id:Union[int, str]) -> None:
        Pose2D.__init__(self, x, y, theta)
        IdentityHeader.__init__(self, id=id)

class Pose2DMarkedStamped(Pose2D, IdentityHeader):
    '''Go to Pose2DMarked and Pose2DStamped.
    '''
    def __init__(self, x:float, y:float, theta:float, id:Union[int, str], timestamp:float) -> None:
        Pose2D.__init__(self, x, y, theta)
        IdentityHeader.__init__(self, id=id, timestamp=timestamp)

#%%# ROS-style more geometric objects
class Polygon2D(ListLike):
    def __init__(self, vertices:List[Point2D]):
        super().__init__(vertices, Point2D)

class Polygon2DStamped(ListLike, IdentityHeader):
    def __init__(self, vertices:List[Point2D], timestamp:float) -> None:
        ListLike.__init__(self, vertices, Point2D)
        IdentityHeader.__init__(self, timestamp=timestamp)

class Polygon2DMarked(ListLike, IdentityHeader):
    def __init__(self, vertices:List[Point2D], id:Union[int, str]) -> None:
        ListLike.__init__(self, vertices, Point2D)
        IdentityHeader.__init__(self, id=id)

class Polygon2DMarkedStamped(ListLike, IdentityHeader):
    def __init__(self, vertices:List[Point2D], id:Union[int, str], timestamp:float) -> None:
        ListLike.__init__(self, vertices, Point2D)
        IdentityHeader.__init__(self, id=id, timestamp=timestamp)

#%%# Objects for path/traj planning
class PathNode(Pose2DMarked):
    def __init__(self, x:float, y:float, theta:float=0.0, id:Union[int, str]=-1) -> None:
        super().__init__(x, y, theta, id)

    def rescale(self, rescale:float):
        self.x = self.x*rescale
        self.y = self.y*rescale

class PathNodeList(ListLike):
    '''
    Two indicators for each PathNode:
    1. Node ID
    2. Node position index
    '''
    def __init__(self, path:List[PathNode]) -> None:
        super().__init__(path, PathNode)
        self.__build_dict()

    def __build_dict(self) -> None:
        '''Build a dictionary so that one can access a node via its the node's ID.
        '''
        self.path_id_dict = {}
        for node in self:
            self.path_id_dict[node.id] = (node.x, node.y, node.theta)

    def get_node_coords(self, node_id)-> tuple:
        '''return based on node id
        '''
        self.__build_dict()
        return self.path_id_dict[node_id]

    def rescale(self, rescale:float) -> None:
        [n.rescale(rescale) for n in self]

class TrajectoryNode(Pose2DStamped):
    def __init__(self, x:float, y:float, theta:float, timestamp:float=-1.0) -> None:
        super().__init__(x, y, theta, timestamp)

    def rescale(self, rescale:float):
        self.x = self.x*rescale
        self.y = self.y*rescale

class TrajectoryNodeList(ListLike):
    def __init__(self, trajectory:List[Union[TrajectoryNode, PathNode]]):
        '''The elements can be TrajectoryNode (better) or PathNode (will be converted to TrajectoryNode).
        '''
        trajectory = [self.__path2traj(x) for x in trajectory]
        super().__init__(trajectory, TrajectoryNode)

    def __path2traj(self, path_node:PathNode) -> TrajectoryNode:
        '''Convert a PathNode into TrajectoryNode (return itself if it is TrajectoryNode already).
        '''
        if isinstance(path_node, TrajectoryNode):
            return path_node
        checktype(path_node, PathNode)
        return TrajectoryNode(path_node.x, path_node.y, path_node.theta)

    def insert(self):
        raise NotImplementedError('No insert method found.')

    def rescale(self, rescale:float) -> None:
        [n.rescale(rescale) for n in self]

#%%# Object to interact with library:nexworkx
class NetGraph(nx.Graph):
    def __init__(self, node_dict:Dict[Any, tuple], edge_list:List[tuple]):
        super.__init__()
        for node_id in node_dict:
            self.add_node(node_id, pos=node_dict[node_id])
        self.add_edges_from(edge_list)

    def set_distance_weight(self):
        def euclidean_distance(graph:nx.Graph, source, target, keyword='pos'):
            x1, y1 = graph.nodes[source][keyword]
            x2, y2 = graph.nodes[target][keyword]
            return math.sqrt((x1-x2)**2 + (y1-y2)**2) 
        for e in self.edges():
            self[e[0]][e[1]]['weight'] = euclidean_distance(self, e[0], e[1])

    def get_x(self, node_id):
        return self.nodes[node_id]['pos'][0]

    def get_y(self, node_id):
        return self.nodes[node_id]['pos'][1]

    def get_node(self, node_id) -> PathNode:
        return PathNode(self.get_x(node_id), self.get_y(node_id), node_id)

    def get_node_pos(self, node_id) -> tuple:
        return self.get_x(node_id), self.get_y(node_id)

    def get_all_edge_positions(self) -> List[List[tuple]]:
        edge_positions = []
        for e in self.edges:
            edge_positions.append([self.get_node_pos(e[0]), self.get_node_pos(e[1])])
        return edge_positions

    def return_given_path(self, path_node_ids:list) -> PathNodeList:
        return PathNodeList([self.get_node(id) for id in path_node_ids])

    def return_random_path(self, start_node_id, num_traversed_nodes:int) -> PathNodeList:
        '''Return random PathNodeList without repeat nodes
        '''
        path_ids = [start_node_id]
        path = PathNodeList([self.get_node(start_node_id)])
        for _ in range(num_traversed_nodes):
            connected_node_ids = list(self.adj[path_ids[-1]])
            connected_node_ids = [x for x in connected_node_ids if x not in path_ids]
            if not connected_node_ids:
                return path
            next_id = random.choice(connected_node_ids) # NOTE: Change this to get desired path pattern
            path_ids.append(next_id)
            path.append(self.get_node(next_id))
        return path


if __name__ == '__main__':
    node_1  = PathNode(2,3,0)
    node_11 = PathNode(3,4,1)
    node_12 = PathNode(4,4,1)
    node_2  = PathNode(5,4,1)
    path_list = [node_1]
    traj_list = [node_1, node_11]

    path = PathNodeList(path_list)
    path.append(node_2)
    traj = TrajectoryNodeList(traj_list)
    traj.append(TrajectoryNode(node_2.x, node_2.y, node_2.theta))
    print(path)
    print(traj.numpy())

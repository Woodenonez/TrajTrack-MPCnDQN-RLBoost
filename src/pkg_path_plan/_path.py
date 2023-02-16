import math

import numpy as np

from typing import Union, List, Tuple


def checktype(object_:object, desired_type:Union[type, List[type]]) -> object:
    '''Check if the given object has the desired type (if so, return the object).
    '''
    if isinstance(desired_type, type):
        if not isinstance(object_, desired_type):
            raise TypeError(f'Input must be a {desired_type}, got {type(object_)}.')
    elif isinstance(desired_type, List):
        [checktype(x, type) for x in desired_type]
        [checktype(object_, x) for x in desired_type]
    else:
        raise TypeError('Desired type must be a type or a list of types.')
    return object_


class ListLike(list):
    """A list-like object is a list of elements, where each element can be converted into a tuple.
    
    An element should have the __call__ method which returns a tuple.
    """
    def __init__(self, input_list:list, element_type:Union[type, List[type]]) -> None:
        super().__init__(input_list)
        self._elem_type = element_type
        self._input_validation(input_list, element_type)

    def _input_validation(self, input_list, element_type):
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
        super().append(checktype(element, self._elem_type))

    def insert(self, position:int, element) -> None:
        super().insert(position, checktype(element, self._elem_type))

    def numpy(self) -> np.ndarray:
        '''
        Description
            :Convert list to np.ndarray. Elements must have __call__ method.
        '''
        return np.array([x() for x in self])


class PathNode:
    def __init__(self, x:float, y:float, theta:float=0.0, id_:Union[int, str]=-1) -> None:
        """ If id_=-1, the node doesn't belong to any graphs.
        """
        self.x = x
        self.y = y
        self.theta = theta
        self._id = id_

    @property
    def id_(self) -> Union[int, str]:
        return self._id

    def __str__(self) -> str:
        return f"{self.__class__.__name__} ({self.x}, {self.y}, {self.theta})"

    def __call__(self) -> tuple:
        return (self.x, self.y, self.theta)

    def __getitem__(self, idx:int) -> float:
        return (self.x, self.y, self.theta)[idx]

    def __eq__(self, other: 'PathNode') -> bool:
        return self.x == other.x and self.y == other.y and self.theta == other.theta

    def __sub__(self, other: 'PathNode') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
    
    def rescale(self, rescale:float):
        self.x = self.x*rescale
        self.y = self.y*rescale


class PathNodeList(ListLike):
    """
    Two indicators for each PathNode:
    1. Node ID
    2. Node position index
    """
    def __init__(self, path: List[PathNode]) -> None:
        super().__init__(path, PathNode)
        self.__build_dict()

    def __build_dict(self) -> None:
        """Build a dictionary so that one can access a node via its the node's ID.
        """
        self.path_id_dict = {}
        for node in self:
            node: PathNode
            self.path_id_dict[node.id_] = (node.x, node.y, node.theta)

    @classmethod
    def from_tuples(cls, list_of_tuples: List[tuple]):
        if len(list_of_tuples[0]) < 2:
            raise ValueError("Input tuples don't have enough elements, at least 2.")
        elif len(list_of_tuples[0]) == 2:
            list_of_nodes = [PathNode(x[0], x[1]) for x in list_of_tuples]
        elif len(list_of_tuples[0]) == 3:
            list_of_nodes = [PathNode(x[0], x[1], x[2]) for x in list_of_tuples]
        else:
            list_of_nodes = [PathNode(x[0], x[1], x[2], x[3]) for x in list_of_tuples]
        return cls(list_of_nodes)

    def get_node_coords(self, node_id)-> Tuple[float, float, float]:
        """return based on node id"""
        self.__build_dict()
        return self.path_id_dict[node_id]

    def rescale(self, rescale: float) -> None:
        [n.rescale(rescale) for n in self]


class TrajectoryNode(PathNode):
    def __init__(self, x:float, y:float, theta:float, id_:Union[int, str]=-1, timestamp:float=0.0):
        super().__init__(x, y, theta, id_)
        self._timestamp = timestamp

    @property
    def timestamp(self) -> float:
        return self._timestamp


class TrajectoryNodeList(ListLike):
    def __init__(self, trajectory:List[Union[TrajectoryNode, PathNode]]):
        """The elements can be TrajectoryNode or PathNode (will be converted to TrajectoryNode).
        """
        trajectory = [self._path2traj(x) for x in trajectory]
        super().__init__(trajectory, TrajectoryNode)

    def _path2traj(self, path_node:PathNode) -> TrajectoryNode:
        """Convert a PathNode into TrajectoryNode (return itself if it is TrajectoryNode already).
        """
        if isinstance(path_node, TrajectoryNode):
            return path_node
        checktype(path_node, PathNode)
        return TrajectoryNode(path_node.x, path_node.y, path_node.theta, path_node.id_)

    @classmethod
    def from_tuples(cls, list_of_tuples: List[tuple]):
        if len(list_of_tuples[0]) < 3:
            raise ValueError("Input tuples don't have enough elements, at least 3.")
        elif len(list_of_tuples[0]) == 3:
            list_of_nodes = [TrajectoryNode(x[0], x[1], x[2]) for x in list_of_tuples]
        elif len(list_of_tuples[0]) == 4:
            list_of_nodes = [TrajectoryNode(x[0], x[1], x[2], x[3]) for x in list_of_tuples]
        else:
            list_of_nodes = [TrajectoryNode(x[0], x[1], x[2], x[3], x[4]) for x in list_of_tuples]
        return cls(list_of_nodes)

    def insert(self):
        raise NotImplementedError('No insert method found.')

    def rescale(self, rescale:float) -> None:
        [n.rescale(rescale) for n in self]

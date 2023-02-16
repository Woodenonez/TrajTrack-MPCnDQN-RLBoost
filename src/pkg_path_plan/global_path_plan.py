import networkx as nx
from .path_plan_graph import dijkstra

from ._path import PathNode, PathNodeList

from typing import Union, Tuple, List

class GlobalPathPlanner:
    """Set the path manually or compute one.
    """
    def __init__(self, graph:nx.Graph) -> None:
        self.G = graph
        self.reset()

    @property
    def global_path(self):
        return self._global_path

    def reset(self):
        self._global_path = None
        self.start_node = None
        self.next_node  = None
        self.final_node = None

    def set_path(self, path: PathNodeList):
        self.__next_node_position = 0
        self._global_path = path
        self.next_node  = path[0]
        self.final_node = path[-1]
        if self.start_node is not None:
            self._global_path.insert(0, self.start_node)

    def set_start_node(self, start: PathNode):
        self.start_node = start
        if self._global_path is not None:
            self._global_path.insert(0, self.start_node)

    def move_to_next_node(self):
        if self.__next_node_position < len(self._global_path)-1:
            self.__next_node_position += 1
            self.next_node = self._global_path[self.__next_node_position]
        else:
            self.__next_node_position = len(self._global_path)-1
            self.next_node = self._global_path[self.__next_node_position]

    def get_shortest_path(self, source, target, algorithm:str='dijkstra'):
        if algorithm == 'dijkstra':
            planner = dijkstra.DijkstraPathPlanner(self.G)
            _, paths = planner.k_shortest_paths(source, target, k=1)
            shortest_path = paths[0]
        else:
            raise NotImplementedError(f"Algorithm {algorithm} is not implemented.")
        self.set_path(shortest_path)

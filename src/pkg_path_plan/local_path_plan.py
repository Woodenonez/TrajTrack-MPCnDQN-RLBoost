
from ._path import PathNodeList

from .path_plan_cspace import visibility

from typing import Union, List


class LocalPathPlanner:
    def __init__(self, graph_map, verbose=False):
        """
        Descriptions:
            The local planner takes start, end and map as inputs, and outputs local path.
        Args:
            graph_map: `Geometric map` or `Occupancy grid map` object
                        :`Geometric map` has "boundary_coords", "obstacle_list", and processed ones.
                        :`Occupancy grid map` has pixel-wize maps in `np.ndarry`.
        Attrs:
            path_planner: Object with the selected path finder algorithm, must have "get_ref_path" method.
            ref_path: Reference path
        """
        input_boundary = graph_map.processed_boundary_coords
        input_obstacle_list = graph_map.processed_obstacle_list
        
        self.path_planner = visibility.VisibilityPathFinder(input_boundary, 
                                                            input_obstacle_list, 
                                                            verbose=verbose)

    def get_ref_path(self, start:tuple, end:tuple) -> PathNodeList:
        ref_path = self.path_planner.get_ref_path(start, end)
        self.ref_path = PathNodeList.from_tuples(ref_path)
        return self.ref_path


import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Callable
from matplotlib.axes import Axes


class GeometricMap:
    """With boundary and obstacle coordinates."""
    def __init__(self, boundary_coords:List[tuple], obstacle_list:List[List[tuple]], inflator:Callable=None):
        """
        Args:
            boundary_coords: A list of tuples, each tuple is a pair of coordinates.
            obstacle_list: A list of lists of tuples, each tuple is a pair of coordinates.
            inflator: A function that inflates a polygon.
        """
        boundary_coords, obstacle_list = self.__input_validation(boundary_coords, obstacle_list)
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        if inflator is not None:
            self.processed_boundary_coords = inflator(boundary_coords)
            self.processed_obstacle_list = [inflator(x) for x in obstacle_list]
        else:
            self.processed_boundary_coords = None
            self.processed_obstacle_list = None

    def __input_validation(self, boundary_coords, obstacle_list):
        if not isinstance(boundary_coords, list):
            raise TypeError('A map boundary must be a list of tuples.')
        if not isinstance(obstacle_list, list):
            raise TypeError('A map obstacle list must be a list of lists of tuples.')
        if len(boundary_coords[0])!=2 or len(obstacle_list[0][0])!=2:
            raise TypeError('All coordinates must be 2-dimension.')
        return boundary_coords, obstacle_list

    def __call__(self, inflated:bool=True) -> Tuple[List[tuple], List[List[tuple]]]:
        if inflated:
            if self.processed_boundary_coords is None or self.processed_obstacle_list is None:
                raise ValueError('No inflated map available.')
            return self.processed_boundary_coords, self.processed_obstacle_list
        return self.boundary_coords, self.obstacle_list

    def get_occupancy_map(self, rescale:int=100) -> np.ndarray:
        """
        Args:
            rescale: The resolution of the occupancy map.
        Returns:
            A numpy array of shape (height, width, 3).
        """
        if not isinstance(rescale, int):
            raise TypeError(f'Rescale factor must be int, got {type(rescale)}.')
        assert(0<rescale<2000),(f'Rescale value {rescale} is abnormal.')
        boundary_np = np.array(self.boundary_coords)
        width  = max(boundary_np[:,0]) - min(boundary_np[:,0])
        height = max(boundary_np[:,1]) - min(boundary_np[:,1])

        fig, ax = plt.subplots(figsize=(width, height), dpi=rescale)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.plot(np.array(self.boundary_coords)[:,0], np.array(self.boundary_coords)[:,1], 'w-')
        for coords in self.obstacle_list:
            x, y = np.array(coords)[:,0], np.array(coords)[:,1]
            plt.fill(x, y, color='k')
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        occupancy_map = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        occupancy_map = occupancy_map.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return occupancy_map

    def plot(self, ax: Axes, inflated:bool=True, original_plot_args:dict={'c':'k'}, inflated_plot_args:dict={'c':'r'}):
        if inflated:
            if self.processed_boundary_coords is None or self.processed_obstacle_list is None:
                raise ValueError('No inflated map available.')
            else:
                plot_boundary = np.array(self.processed_boundary_coords+[self.processed_boundary_coords[0]])
                ax.plot(plot_boundary[:,0], plot_boundary[:,1], **inflated_plot_args)
                for coords in self.processed_obstacle_list:
                    plot_obstacle = np.array(coords+[coords[0]])
                    plt.fill(plot_obstacle[:,0], plot_obstacle[:,1], **inflated_plot_args)
        plot_boundary = np.array(self.boundary_coords+[self.boundary_coords[0]])
        ax.plot(plot_boundary[:,0], plot_boundary[:,1], **original_plot_args)
        for coords in self.obstacle_list:
            plot_obstacle = np.array(coords+[coords[0]])
            plt.fill(plot_obstacle[:,0], plot_obstacle[:,1], **original_plot_args)


if __name__ == '__main__':
    boundary = [(0,0), (10,0), (10,10), (0,10)]
    obstacle_list = [[(1,1), (2,1), (2,2), (1,2)], [(3,3), (4,3), (4,4), (3,4)]]
    map = GeometricMap(boundary, obstacle_list)
    map.get_occupancy_map()
    fig, ax = plt.subplots()
    map.plot(ax, inflated=False)
    plt.show()
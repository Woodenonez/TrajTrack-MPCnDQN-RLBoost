from abc import ABC, abstractmethod

from typing import List, Union, Tuple


class ObstacleSimulator(ABC):
    def __init__(self, sampling_time: float, obstacle_radius: float, birth_time:float=0.0, max_prediction_time_offset:int=20) -> None:
        """
        Args:
            sampling_time: The sampling time of the obstacle.
            obstacle_radius: The radius of the obstacle.
            birth_time: The birth time of the obstacle.
            max_prediction_time_offset: The maximum prediction time offset.
        """
        self.ts = sampling_time
        self.r = obstacle_radius
        self.birth_time = birth_time
        self.T_max = max_prediction_time_offset

        self.num_mode:int # fixed

    @abstractmethod
    def _create_obstacle(self, **kwargs) -> None:
        """Create the (multimodal) motion for the obstacle.
        It should be in the form of a list of dictionaries, where the list index is the (relative) time step.
        Each dictionary contains all info of the target at that time step.

        The format of the dictionary is
            :{'info':[kt,x,y,theta], 'pred_T1':[[a1,x1,y1,sx1,sy1,theta1], ..., [am,xm,ym,sxm,sym,thetam]], 'pred_T2':..., ...}
            :where 'm' is the number of components/futures and 'a' is the weight.
            :note that kt is the relative time step (from 0).
        """
        self.obj_list = []

    def get_obs_dict(self, current_time) -> Union[dict, None]:
        """
        Returns:
            :The dictionary of the obstacle information at the current time,
            :or `None` if the obstacle does not exist at the current time.
        """
        if current_time >= self.birth_time:
            time_step = int((current_time-self.birth_time)/self.ts)
            if time_step <= len(list(self.obj_list))-1:
                obs_dict:dict = self.obj_list[time_step]
                return obs_dict
        return None
    
    def get_full_obstacle_list(self, current_time: float, factor:float=1.0) -> list:
        """
        Note:
            `obs_dict`: {'info':[kt,x,y,theta], 'pred_T1':[[a1,x1,y1,sx1,sy1,theta1], ...], 'pred_T2':..., ...}
        Args:
            factor: The factor to scale the obstacle uncertainty.
        Returns:
            The list of the obstacle information.

            For n modes, [obstacle_1, obstacle_2, ..., obstacle_n],
            :where each mode, obstacle = [obstacle_t1, obstacle_t2, ..., obstacle_tn],
            :where each obstacle_tj = (x, y, sx, sy, theta, alpha)
        """
        obs_dict = self.get_obs_dict(current_time) # pred
        if obs_dict is None:
            return []

        obstacle_modes_list = []
        for n in range(self.num_mode):
            obs_list = []
            for T in range(self.T_max):
                try:
                    obs_pred = obs_dict[list(obs_dict)[T+1]][n]
                except: # if there is no prediction for the mode
                    obs_pred = obs_dict[list(obs_dict)[T+1]][0]
                alpha, x, y, sx ,sy, angle = obs_pred
                obs_list.append((x, y, sx*factor+self.r, sy*factor+self.r, angle, alpha)) # adjust the order
            obstacle_modes_list.append(obs_list)
        return obstacle_modes_list

    @DeprecationWarning
    def get_obstacle_info(self, current_time, key):
        """
        Args:
            current_time: The current time.
            key: The key of the information to be returned.
        Returns:
            The information of the obstacle at the current time."""
        pass
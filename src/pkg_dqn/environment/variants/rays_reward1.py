from ..components import *
from .. import MapGenerator, MobileRobot
from ..environment import TrajectoryPlannerEnvironment


class TrajectoryPlannerEnvironmentRaysReward1(TrajectoryPlannerEnvironment):
    """
    Environment with what the associated report describes as ray and sector
    observations and reward R_1
    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.2,
        reference_path_sample_offset: float = 0,
        corner_samples: int = 3,
        use_memory: bool = True,
        num_segments: int = 8,
        collision_reward_factor: float = 4,
        reach_goal_reward_factor: float = 3,
        cross_track_reward_factor: float = 0.05,
        reference_speed: float = MobileRobot().cfg.SPEED_MAX * 0.8,
        path_progress_factor: float = 2
    ):
        super().__init__(
            [
                SpeedObservation(),
                AngularVelocityObservation(),
                ReferencePathSampleObservation(1, 0, reference_path_sample_offset),
                ReferencePathCornerObservation(corner_samples),
                
                SectorAndRayObservation(num_segments, use_memory=use_memory),
                
                CollisionReward(collision_reward_factor),
                CrossTrackReward(cross_track_reward_factor),
                
                ReachGoalReward(reach_goal_reward_factor),
                ExcessiveSpeedReward(2 * path_progress_factor, reference_speed),
                PathProgressReward(path_progress_factor),
            ],
            generate_map,
            time_step
        )

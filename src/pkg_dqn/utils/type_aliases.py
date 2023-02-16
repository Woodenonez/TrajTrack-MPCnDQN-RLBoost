"""
Used by per_dqn.py

See https://github.com/cloudpipe/cloudpickle/issues/460
"""

from typing import NamedTuple

import torch
import numpy as np

from stable_baselines3.common.type_aliases import TensorDict

class PerReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    rewards: torch.Tensor
    indices: np.ndarray
    weights: np.ndarray

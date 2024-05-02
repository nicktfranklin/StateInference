from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
from torch import Tensor

from task.gridworld import ActType, ObsType, OutcomeTuple

ObservationTuple = namedtuple("ObservationTuple", "obs a r next_obs")


class RolloutDataset:
    """
    This class is meant to be consistent with the dataset in d4RL
    """

    def __init__(
        self,
        action: Optional[List[ActType]] = None,
        obs: Optional[List[ObsType]] = None,
        next_obs: Optional[List[ObsType]] = None,
        reward: Optional[List[float]] = None,
        terminated: Optional[List[bool]] = None,
        truncated: Optional[List[bool]] = None,
        info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.action = action if action is not None else []
        self.obs = obs if obs is not None else []
        self.next_obs = next_obs if next_obs is not None else []
        self.reward = reward if reward is not None else []
        self.terminated = terminated if terminated is not None else []
        self.truncated = truncated if truncated is not None else []
        self.info = info if info is not None else []

    def add(self, obs: ObsType, action: ActType, obs_tuple: OutcomeTuple):
        self.action.append(action)
        self.obs.append(obs)
        self.next_obs.append(obs_tuple[0])  # these are sucessor observations
        self.reward.append(obs_tuple[1])
        self.terminated.append(obs_tuple[2])
        self.truncated.append(obs_tuple[3])
        self.info.append(obs_tuple[4])

    def get_dataset(self, n_obs: int | None = None) -> dict[str, Union[Any, Tensor]]:
        """This is meant to be consistent with the dataset in d4RL

        n_obs: (int) pull the last n_obs from the buffer, if None, pull all
        """
        if n_obs is not None:
            obs = self.obs[-n_obs:]
            next_obs = self.next_obs[-n_obs:]
            action = self.action[-n_obs:]
            reward = self.reward[-n_obs:]
            terminated = self.terminated[-n_obs:]
            truncated = self.truncated[-n_obs:]
        else:
            obs = self.obs
            next_obs = self.next_obs
            action = self.action
            reward = self.reward
            terminated = self.terminated
            truncated = self.truncated

        return {
            "observations": np.stack(obs),
            "next_observations": np.stack(next_obs),
            "actions": np.stack(action),
            "rewards": np.stack(reward),
            "terminated": np.stack(terminated),
            "timeouts": np.stack(truncated),  # timeouts are truncated
            "infos": self.info,
        }

    def get_obs(self, idx: int) -> ObservationTuple:
        return ObservationTuple(
            self.obs[idx],
            self.action[idx],
            self.reward[idx],
            self.next_obs[idx],
        )

    def __len__(self) -> int:
        return len(self.obs)

    def reset_buffer(self):
        self.action = []
        self.obs = []
        self.next_obs = []
        self.reward = []
        self.terminated = []
        self.truncated = []
        self.info = []


# TODO: Remove this class
@dataclass
class OaroTuple:
    obs: ObsType
    a: ActType
    r: float
    next_obs: Tensor
    index: int  # unique index for each trial

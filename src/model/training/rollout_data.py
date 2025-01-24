from abc import ABC, abstractmethod
from collections import deque, namedtuple
from heapq import heappop, heappush
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from src.task.gridworld import ActType, ObsType, OutcomeTuple

ObservationTuple = namedtuple("ObservationTuple", "obs a r next_obs")


class BaseBuffer(ABC):
    def __init__(self, capacity: int = None):
        self.capacity = capacity

    @abstractmethod
    def add(self, obs: ObsType, action: ActType, obs_tuple: OutcomeTuple): ...

    @abstractmethod
    def get_dataset(self) -> dict[str, Union[Any, Tensor]]: ...

    @abstractmethod
    def get_obs(self, idx: int) -> ObservationTuple: ...

    @abstractmethod
    def __len__(self): ...

    @abstractmethod
    def get_obs(self, idx: int) -> ObservationTuple: ...

    @abstractmethod
    def reset_buffer(self): ...


class RolloutBuffer(BaseBuffer):
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
        capacity: Optional[int] = None,
        run: Optional[List[int]] = None,
        epoch: Optional[List[int]] = None,
    ) -> None:
        self.action = action if action is not None else []
        self.obs = obs if obs is not None else []
        self.next_obs = next_obs if next_obs is not None else []
        self.reward = reward if reward is not None else []
        self.terminated = terminated if terminated is not None else []
        self.truncated = truncated if truncated is not None else []
        self.info = info if info is not None else []
        self.capcity = capacity
        self.run = run if run is not None else []
        self.epoch = epoch if epoch is not None else []

    def add(
        self,
        obs: ObsType,
        action: ActType,
        obs_tuple: OutcomeTuple,
        run: int = 0,
        epoch: int = 0,
    ):
        self.action.append(action)
        self.obs.append(obs)
        self.next_obs.append(obs_tuple[0])  # these are sucessor observations
        self.reward.append(obs_tuple[1])
        self.terminated.append(obs_tuple[2])
        self.truncated.append(obs_tuple[3])
        self.info.append(obs_tuple[4])
        self.run.append(run)
        self.epoch.append(epoch)

        if self.capcity is not None and len(self.obs) > self.capcity:
            self.action.pop(0)
            self.obs.pop(0)
            self.next_obs.pop(0)
            self.reward.pop(0)
            self.terminated.pop(0)
            self.truncated.pop(0)
            self.info.pop(0)
            self.run.pop(0)

    def get_dataset(self) -> dict[str, Union[Any, Tensor]]:
        """This is meant to be consistent with the dataset in d4RL"""

        return {
            "observations": np.stack(self.obs),
            "next_observations": np.stack(self.next_obs),
            "actions": np.stack(self.action),
            "rewards": np.stack(self.reward),
            "terminated": np.stack(self.terminated),
            "timouts": np.stack(self.truncated),  # timeouts are truncated
            "infos": self.info,
        }

    def get_average_reward(self, epoch: int = 0) -> float:
        return np.mean([r for r, e in zip(self.reward, self.epoch) if e == epoch])

    def get_average_run_length(self, epoch: int = 0) -> float:
        runs = np.stack(self.run)
        epochs = np.stack(self.epoch)

        n_runs = runs[epochs == epoch].max()
        n_timesteps = len(runs[epochs == epoch])
        return n_timesteps / float(n_runs)

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

    def transpose_obs(self, axis_order: List[int]) -> None:
        self.obs = [np.transpose(obs, axis_order) for obs in self.obs]
        self.next_obs = [
            np.transpose(next_obs, axis_order) for next_obs in self.next_obs
        ]


class Episode:

    def __init__(self, aggregation: Callable | None = None):
        self.actions = []
        self.obs = []
        self.next_obs = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.info = []

        # used in the priority buffer
        self.total_reward = 0

        self.aggregation = aggregation if aggregation else np.mean

    def add(self, obs: ObsType, action: ActType, obs_tuple: OutcomeTuple):

        next_obs, reward, terminated, truncated, info = obs_tuple

        self.actions.append(action)
        self.obs.append(obs)
        self.next_obs.append(next_obs)  # these are sucessor observations
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.info.append(info)

        self.total_reward += reward

    def __len__(self):
        return len(self.obs)

    def __lt__(self, obj):
        """used for the min heap"""

        if self.aggregation(self.rewards) == self.aggregation(obj.rewards):
            # logic here is to use a metric of dispersion between the observations
            # as a tie-breaker. I.e. episodes that have a lot of different state
            # are better.  Variance of each dimension is a natural choice, as
            # it is fast to calculate and will penalize the episodes that have
            # cycles.

            def mean_var(obs: List[np.array]):
                return np.stack(obs).reshape(len(obs), -1).var(axis=0).mean()

            return mean_var(self.obs) < mean_var(obj.obs)

        return self.aggregation(self.rewards) < self.aggregation(obj.rewards)

    @property
    def is_terminated(self) -> bool:
        return self.terminated[-1] or self.truncated[-1]

    def get_dataset(self) -> dict[str, Union[Any, Tensor]]:
        """This is meant to be consistent with the dataset in d4RL. Note,
        this is not ordered consistent with the visitation order"""

        return {
            "observations": np.stack(self.obs),
            "next_observations": np.stack(self.next_obs),
            "actions": np.stack(self.actions),
            "rewards": np.stack(self.rewards),
            "terminated": np.stack(self.terminated),
            "timouts": np.stack(self.truncated),  # timeouts are truncated
            "infos": self.info,
        }


class PriorityReplayBuffer(BaseBuffer):
    ### use a min queue

    def __init__(
        self, capacity: int | None = None, aggregation: Callable | None = None
    ):
        self.capacity = capacity
        self.buffer_size = 0

        self.current_episode = None
        self.min_heap = []
        self.aggregation = aggregation

    def _store_episode(self, episode: Episode) -> None:
        heappush(self.min_heap, episode)

        if (
            self.capacity is not None and self.buffer_size > self.capacity
        ):  # remove an episode with low reward
            episode_to_remove = heappop(self.min_heap)

            # account for the change in buffer size
            self.buffer_size -= len(episode_to_remove)

    def add(self, obs: ObsType, action: ActType, obs_tuple: OutcomeTuple):

        if self.current_episode is None:
            self.current_episode = Episode(self.aggregation)

        self.current_episode.add(obs, action, obs_tuple)
        self.most_recent_episode = self.current_episode
        self.buffer_size += 1

        if self.current_episode.is_terminated:
            # add the new episode to the heap (only adds completed episodes)
            self._store_episode(self.current_episode)

            # we never want an empty episode
            self.current_episode = None
            # # make a new episode. Note, we don't add the current
            # # episode to the heap to (1) maintain the heap property
            # # and (2) to prevent it from being removed if the
            # # capacity is exceeded
            # self.current_episode = Episode()

    def reset_buffer(self):
        self.buffer_size = 0
        self.current_episode = None
        self.min_heap = []

    def __len__(self) -> int:
        return self.buffer_size

    def get_obs(self, idx: int) -> ObservationTuple:
        """Use with caution: this ordering will change as the buffer grows"""

        #
        assert idx >= 0, "negative indexing not supported"
        t = idx

        for episode in self.min_heap:
            if t >= len(episode):
                t -= len(episode)
            else:
                return ObservationTuple(
                    episode.obs[t],
                    episode.actions[t],
                    episode.rewards[t],
                    episode.next_obs[t],
                )
        return ObservationTuple(
            self.current_episode.obs[t],
            self.current_episode.actions[t],
            self.current_episode.rewards[t],
            self.current_episode.next_obs[t],
        )

    def get_dataset(self) -> dict[str, Union[Any, Tensor]]:
        """This is meant to be consistent with the dataset in d4RL. Note,
        this is not ordered consistent with the visitation order"""

        obs, next_obs, actions, rewards, terminated, truncated = [], [], [], [], [], []
        infos = []

        episode: Episode

        for episode in self.iterator():
            obs.extend(episode.obs)
            next_obs.extend(episode.next_obs)
            actions.extend(episode.actions)
            rewards.extend(episode.rewards)
            terminated.extend(episode.terminated)
            truncated.extend(episode.truncated)
            infos.extend(episode.info)

        return {
            "observations": np.stack(obs),
            "next_observations": np.stack(next_obs),
            "actions": np.stack(actions),
            "rewards": np.stack(rewards),
            "terminated": np.stack(terminated),
            "timouts": np.stack(truncated),  # timeouts are truncated
            "infos": infos,
        }

    def iterator(self):
        return iter(self.min_heap)

    def transpose_obs(self, axis_order: List[int]) -> None:
        for episode in self.iterator():
            episode.obs = [np.transpose(obs, axis_order) for obs in episode.obs]
            episode.next_obs = [
                np.transpose(next_obs, axis_order) for next_obs in episode.next_obs
            ]


class EpisodeBuffer(PriorityReplayBuffer):

    def __init__(self, capacity: int | None = None, **kwargs):
        self.capacity = capacity
        self.buffer_size = 0
        self.current_episode = None
        self.queue = deque()

    def _store_episode(self, episode: Episode) -> None:
        self.queue.append(episode)

        if (
            self.capacity is not None and self.buffer_size > self.capacity
        ):  # remove an episode with low reward
            episode_to_remove = self.queue.popleft()

            # account for the change in buffer size
            self.buffer_size -= len(episode_to_remove)

    def reset_buffer(self):
        self.buffer_size = 0
        self.current_episode = None
        self.queue = deque()

    def iterator(self):
        return iter(self.queue)


class PpoEpisode(Episode):
    def __init__(self, aggregation: Callable | None = None):
        self.actions = []
        self.obs = []
        self.next_obs = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.info = []
        self.log_probs = []
        self.values = []

        # used in the priority buffer
        self.total_reward = 0

        self.aggregation = aggregation if aggregation else np.mean

    def add(
        self,
        obs: ObsType,
        action: ActType,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        obs_tuple: OutcomeTuple,
    ):

        next_obs, reward, terminated, truncated, info = obs_tuple

        self.actions.append(action)
        self.obs.append(obs)
        self.next_obs.append(next_obs)  # these are sucessor observations
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.info.append(info)
        self.log_probs.append(log_probs)
        self.values.append(values)

        self.total_reward += reward

    def get_dataset(self) -> dict[str, Union[Any, Tensor]]:
        """This is no longer consistent with the dataset in d4RL. Note,
        this is not ordered consistent with the visitation order"""

        return {
            "observations": np.stack(self.obs),
            "next_observations": np.stack(self.next_obs),
            "actions": np.stack(self.actions),
            "rewards": np.stack(self.rewards),
            "values": torch.stack(self.values),
            "terminated": np.stack(self.terminated),
            "timouts": np.stack(self.truncated),  # timeouts are truncated
            "infos": self.info,
            "log_probs": torch.stack(self.log_probs),
        }


class PpoBuffer(EpisodeBuffer):

    def add(
        self,
        obs: ObsType,
        action: ActType,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        obs_tuple: OutcomeTuple,
    ):

        if self.current_episode is None:
            self.current_episode = PpoEpisode()

        self.current_episode.add(obs, action, values, log_probs, obs_tuple)
        self.most_recent_episode = self.current_episode
        self.buffer_size += 1

        if self.current_episode.is_terminated:
            # add the new episode to the heap (only adds completed episodes)
            self._store_episode(self.current_episode)

            # we never want an empty episode
            self.current_episode = None

    def finish(self):
        self._store_episode(self.current_episode)
        self.current_episode = None

    def get_dataset(self) -> Dict[str, Any | Tensor]:
        raise NotImplementedError("Ppo Buffer doesnt support this method")

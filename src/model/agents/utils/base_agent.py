import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Iterable, List, Optional, Union

import gymnasium as gym
import numpy as np
import pytorch_lightning as pl
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ConvertCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.vec_env import VecEnv
from torch import FloatTensor, Tensor
from tqdm import tqdm

from src.model.training.rollout_data import (
    BaseBuffer,
    PriorityReplayBuffer,
    RolloutBuffer,
)
from src.task.gridworld import ActType, ObsType
from src.utils.sampling_functions import inverse_cmf_sampler

from .tabular_agent_pytorch import ModelBasedAgent


class BaseAgent(ABC):
    def __init__(self, task: VecEnv | gym.Env) -> None:
        super().__init__()
        self.task = task if isinstance(task, gym.Env) else task.envs[0]
        self.num_timesteps = 0

    def collocate(
        self,
        x: (
            Union[torch.tensor, np.ndarray]
            | Dict[str, Union[torch.tensor, np.ndarray, Any]]
            | Any
        ),
    ) -> torch.tensor:
        if isinstance(x, dict):
            return {k: self.collocate(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [self.collocate(v) for v in x]

        if isinstance(x, np.ndarray):
            if x.dtype == np.float64:
                x = x.astype(np.float32)
            return torch.tensor(x, device=self.device)

        if torch.is_tensor(x):
            if x.dtype == torch.float64:
                x = x.float()
            return x.to(self.device)

        return x

    @abstractmethod
    def get_pmf(self, x: FloatTensor) -> np.ndarray: ...

    @abstractmethod
    def get_value_fn(self, x: FloatTensor) -> FloatTensor: ...

    def get_env(self) -> VecEnv:
        ## used for compatibility with stablebaseline code, use with caution
        if isinstance(self.task, VecEnv):
            return self.task
        return BaseAlgorithm._wrap_env(self.task, verbose=False, monitor_wrapper=True)

    def get_task(self) -> gym.Env:
        return self.task if isinstance(self.task, gym.Env) else self.task[0]

    def save(self, model_path: str):
        """this is a stable baselines method we don't use"""
        pass

    @abstractmethod
    def predict(
        self,
        obs: ObsType,
        state: Optional[FloatTensor] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = False,
    ) -> tuple[ActType, Optional[FloatTensor]]: ...

    def _init_state(self) -> Optional[FloatTensor]:
        return None

    def update_rollout_policy(
        self,
        obs: int,
        a: int,
        outcome_tuple,
        rollout_buffer: BaseBuffer,
    ) -> None:
        pass

    def update_from_batch(self, batch: BaseBuffer, progress_bar: bool = False):
        pass

    def collect_rollouts(
        self,
        n_rollout_steps: int,
        rollout_buffer: BaseBuffer,
        epoch: int = 0,
    ) -> bool:
        task = self.get_task()
        obs = task.reset()[0]
        state = self._init_state()
        episode_start = True

        run_id = 0
        for _ in tqdm(range(n_rollout_steps), desc="Collection rollouts"):
            action, state = self.predict(obs, state, episode_start, deterministic=False)
            episode_start = False

            outcome_tuple = task.step(action)
            self.num_timesteps += 1
            rollout_buffer.add(obs, action, outcome_tuple, run_id, epoch)

            self.update_rollout_policy(obs, action, outcome_tuple, rollout_buffer)

            # get the next obs from the observation tuple
            obs, reward, done, truncated, _ = outcome_tuple

            if done or truncated:
                obs = task.reset()[0]
                run_id += 1

        return True

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
        tensorboard: bool = True,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def learn(
        self,
        total_timesteps: int,
        progress_bar: bool = False,
        reset_buffer: bool = False,
        capacity: Optional[int] = None,
        callback: MaybeCallback = None,
        buffer_class: str | None = None,
        buffer_kwargs: Dict[str, Any] | None = None,
        **kwargs,
    ):
        logging.info("Calling Library learn method")
        if buffer_class is None or buffer_class == "fifo":
            self.rollout_buffer = RolloutBuffer(capacity=capacity)
        elif buffer_class == "priority":
            buffer_kwargs = buffer_kwargs if buffer_kwargs else dict()
            self.rollout_buffer = PriorityReplayBuffer(
                capacity=capacity, **buffer_kwargs
            )
        else:
            raise ValueError(f"Buffer class: '{buffer_class}' not implemented!")

        callback = self._init_callback(callback, progress_bar=progress_bar)
        callback.on_training_start(locals(), globals())

        # alternate between collecting rollouts and batch updates
        n_rollout_steps = self.n_steps if self.n_steps is not None else total_timesteps

        self.num_timesteps = 0
        while self.num_timesteps < total_timesteps:
            if reset_buffer:
                self.rollout_buffer.reset_buffer()

            n_rollout_steps = min(n_rollout_steps, total_timesteps - self.num_timesteps)

            if not self.collect_rollouts(
                n_rollout_steps,
                self.rollout_buffer,
                callback=callback,
                progress_bar=progress_bar,
            ):
                break

            self.update_from_batch(self.rollout_buffer, progress_bar=False)

        callback.on_training_end()

    def get_policy_prob(
        self, env: VecEnv, n_states: int, map_height: int, cnn: bool = True
    ) -> FloatTensor:
        """
        Wrapper for getting the policy probability for each state in the environment.
        Requires a gridworld environment, and samples an observation from each state.

        Returns a tensor of shape (n_states, n_actions)

        :param env:
            :param n_states:
            :param map_height:
        """
        assert isinstance(env, VecEnv)
        env = env.envs[0]

        # reshape to match env standard (HxWxC) -> not standard
        shape = [map_height, map_height]
        if cnn:
            shape = [map_height, map_height, 1]

        obs = [
            torch.tensor(env.unwrapped.generate_observation(s)).view(*shape)
            for s in range(n_states)
        ]
        obs = torch.stack(obs)
        with torch.no_grad():
            return self.get_pmf(obs)

    def get_value_function(
        self, env: VecEnv, n_states: int, map_height: int, cnn: bool = True
    ) -> FloatTensor:
        """
        Wrapper for getting the value function for each state in the environment.
        Requires a gridworld environment, and samples an observation from each state.

        Returns a tensor of shape (n_states, 1)

        :param env:
            :param n_states:
            :param map_height:
        """
        assert isinstance(env, VecEnv)
        env = env.envs[0]

        # reshape to match env standard (HxWxC) -> not standard
        shape = [map_height, map_height]
        if cnn:
            shape = [map_height, map_height, 1]

        obs = [
            torch.tensor(env.unwrapped.generate_observation(s)).view(*shape)
            for s in range(n_states)
        ]
        obs = torch.stack(obs)
        with torch.no_grad():
            return self.get_value_fn(obs)

    # todo: implement a progressbar, max steps, etc.  Should be an inplace method
    # look to the BaseAgent method for inspiration
    def collect_buffer(
        self,
        task: gym.Env,
        buffer: BaseBuffer,
        n: int,
        epsilon: float = 0.05,
        start_state: Optional[int] = None,
    ):
        # collect data

        # remove options for stable baselines
        # options = {"initial_state": start_state} if start_state is not None else None
        obs = task.reset()[0]
        done = False
        self.eval()

        for _ in tqdm(range(n), desc="Collection rollouts"):
            action_pmf = self.get_pmf(
                torch.tensor(obs, device=self.device).unsqueeze(0)
            )

            # epsilon greedy
            action_pmf = (1 - epsilon) * action_pmf + epsilon * np.ones_like(
                action_pmf
            ) / len(action_pmf)

            # sample
            action = inverse_cmf_sampler(action_pmf)

            outcome_tuple = task.step(action)
            buffer.add(obs, action, outcome_tuple)

            obs = outcome_tuple[0]
            done = outcome_tuple[2]

            if done:
                obs = task.reset()[0]

        return buffer

    def get_graph_laplacian(
        self,
        rollout_buffer: BaseBuffer,
        normalized: bool = True,
        terminal_state: str = "terminal",
    ) -> tuple[np.ndarray, Dict[Hashable, int]]:

        dataset = rollout_buffer.get_dataset()
        mdp = self.estimate_world_model(rollout_buffer)
        return mdp.get_graph_laplacian(
            normalized=normalized, terminal_state=terminal_state
        )

    def estimate_world_model(
        self,
        rollout_buffer: BaseBuffer,
        gamma: float = 0.95,
    ) -> ModelBasedAgent:
        dataset = rollout_buffer.get_dataset()

        states = self.get_states(self.collocate(dataset["observations"]))
        next_states = self.get_states(self.collocate(dataset["next_observations"]))

        n_states = torch.cat([states, next_states]).unique().shape[0]

        mdp = ModelBasedAgent(
            n_states,
            self.get_env().action_space.n,
            gamma,
        )

        for s, a, r, sp, done in zip(
            states,
            dataset["actions"],
            dataset["rewards"],
            next_states,
            dataset["terminated"],
        ):
            mdp.update(s.item(), a.item(), r.item(), sp.item(), done.item())

        mdp.estimate()

        return mdp

    def dehash_states(
        self, rollout_buffer: BaseBuffer, gamma: float = 0.99
    ) -> np.ndarray:
        """this require a defined value model or estimator"""
        raise NotImplementedError

    def get_states(self, obs: Tensor) -> List[Hashable]:
        raise NotImplementedError


class BaseVaeAgent(BaseAgent, ABC, pl.LightningModule):
    log_kwargs = {
        "prog_bar": False,
        "on_step": True,
        "on_epoch": True,
        "sync_dist": True,
    }

    def __init__(
        self,
        task,
        buffer_capacity: int | None = None,
        rollout_len: int = 2048,
        total_timesteps: int = 2048,
        reset_buffer: bool = False,
    ):
        super().__init__(task)
        self.buffer_capacity = (
            buffer_capacity if buffer_capacity is not None else self.n_steps
        )
        self.n_steps = rollout_len
        self.total_timesteps = total_timesteps
        self.rollout_buffer = self.initialize_buffer()
        self.reset_buffer = reset_buffer
        self.num_timesteps = 0
        # self.global_step = 0

    @abstractmethod
    def update_from_batch(self, batch: BaseBuffer): ...

    @abstractmethod
    def get_states(self, obs: Tensor) -> Hashable: ...

    def initialize_buffer(self):
        return RolloutBuffer(capacity=self.buffer_capacity)

    def training_step(self, batch=None, batch_idx=None):
        """This corresponds to a single epoch of training."""

        n_rollout_steps = min(self.n_steps, self.total_timesteps - self.num_timesteps)

        # collect rollouts
        if not self.collect_rollouts(n_rollout_steps, self.rollout_buffer):
            return

        self.log("train/num_timesteps", self.num_timesteps, **self.log_kwargs)
        self.log(
            "train/Average Reward",
            self.rollout_buffer.get_average_reward(),
            **self.log_kwargs,
        )
        self.log(
            "train/Average Episode Length",
            self.rollout_buffer.get_average_run_length(),
            **self.log_kwargs,
        )

        # update the model
        self.update_from_batch(self.rollout_buffer)

    def validation_step(self, batch=None, batch_idx=None):
        """This corresponds to a single epoch of validation."""
        pass

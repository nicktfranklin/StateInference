import logging
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import gymnasium as gym
import numpy as np
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
from torch import FloatTensor
from tqdm import tqdm

from model.training.rollout_data import RolloutDataset
from task.gridworld import ActType, ObsType
from utils.sampling_functions import inverse_cmf_sampler


class BaseAgent(ABC):
    def __init__(self, task: VecEnv | gym.Env) -> None:
        super().__init__()
        self.env = task if isinstance(task, gym.Env) else task.envs[0]
        self.num_timesteps = 0

        self.observation_space = task.observation_space
        self.action_space = task.action_space

    @abstractmethod
    def get_pmf(self, x: FloatTensor) -> FloatTensor: ...

    def get_env(self) -> VecEnv:
        ## used for compatibility with stablebaseline code, use with caution
        if isinstance(self.env, VecEnv):
            return self.env
        return BaseAlgorithm._wrap_env(self.env, verbose=False, monitor_wrapper=True)

    def get_task(self) -> gym.Env:
        return self.env if isinstance(self.env, gym.Env) else self.envs[0]

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

    def update_rollout_policy(self, rollout_buffer: RolloutDataset) -> None:
        pass

    def collect_rollouts(
        self,
        n_rollout_steps: int,
        rollout_buffer: RolloutDataset,
        callback: BaseCallback,
        progress_bar: Iterable | bool = False,
    ) -> bool:
        task = self.get_task()
        obs = task.reset()[0]
        state = self._init_state()
        episode_start = True

        callback.on_rollout_start()

        for _ in range(n_rollout_steps):
            action, state = self.predict(obs, state, episode_start, deterministic=False)
            episode_start = False

            outcome_tuple = task.step(action)
            self.num_timesteps += 1
            rollout_buffer.add(obs, action, outcome_tuple)

            self.update_rollout_policy(rollout_buffer)

            # get the next obs from the observation tuple
            obs, reward, done, truncated, _ = outcome_tuple

            if done or truncated:
                obs = task.reset()[0]

            callback.update_locals(locals())
            if not callback.on_step():
                return False

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    @abstractmethod
    def update_from_batch(self, batch: RolloutDataset): ...

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
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
        reset_buffer: bool = True,
        callback: MaybeCallback = None,
        **kwargs,
    ):
        logging.info("Calling Library learn method")

        self.rollout_buffer = RolloutDataset()

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

    # todo: implement a progressbar, max steps, etc.  Should be an inplace method
    # look to the BaseAgent method for inspiration
    def collect_buffer(
        self,
        task: gym.Env,
        buffer: RolloutDataset,
        n: int,
        epsilon: float = 0.05,
        start_state: Optional[int] = None,
    ):
        # collect data
        options = {"initial_state": start_state} if start_state is not None else None
        obs = task.reset(options=options)[0]
        done = False

        for _ in tqdm(range(n), desc="Collection rollouts"):
            action_pmf = self.get_pmf(torch.tensor(obs).unsqueeze(0))

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
                obs = task.reset(options=options)[0]

        return buffer

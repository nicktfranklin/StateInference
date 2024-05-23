from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import FloatTensor, Tensor

import model.state_inference.vae
from model.agents.base_state_agent import BaseStateAgent
from model.state_inference.vae import StateVae
from model.training.rollout_data import RolloutDataset
from task.utils import ActType


class DiscretePPO(BaseStateAgent):

    def __init__(
        self,
        env: gym.Env,
        state_inference_model: StateVae,
        policy: ActorCriticPolicy,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(env, state_inference_model, optim_kwargs)
        self.get_policy = policy

    def get_policy(self, obs: Tensor):
        raise NotImplementedError()

    def get_pmf(self, obs: FloatTensor) -> np.ndarray:
        raise NotImplementedError()

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        raise NotImplementedError()

    def update_from_batch(self, buffer: RolloutDataset, progress_bar: bool = False):
        """
        Pseudoe code steps for the inner loop:

        1) Compute Rewards to go
        2) Compute Advantage based on current value function
        3) Update the policy by maximizing the PPO loss

        4) Update the value function by minimizing the value loss"""
        # 1) Compute Rewards to go
        rewards_to_go = self.compute_rewards_to_go(buffer)

        # 2) Compute Advantage based on current value function
        advantages = self.compute_advantages(buffer)

        # 3) Update the policy by maximizing the PPO loss
        self.update_policy(buffer, advantages)

        # 4) Update the value function by minimizing the value loss
        self.update_value_function(buffer, rewards_to_go)

        raise NotImplementedError()

    def learn(
        self,
        total_timesteps: int,
        progress_bar: bool = False,
        reset_buffer: bool = False,
        callback: BaseCallback | None = None,
    ):
        super().learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            reset_buffer=reset_buffer,
            callback=callback,
        )

    @classmethod
    def make_from_configs(
        cls,
        task,
        agent_config: Dict[str, Any],
        vae_config: Dict[str, Any],
        env_kwargs: Dict[str, Any],
    ):
        raise NotImplementedError
        VaeClass = getattr(model.state_inference.vae, agent_config["vae_model_class"])
        vae = VaeClass.make_from_configs(vae_config, env_kwargs)
        return cls(task, vae, **agent_config["state_inference_model"])

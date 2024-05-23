from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import gymnasium as gym
import torch
from torch import Tensor

from model.agents.utils.base_agent import BaseAgent
from model.agents.utils.state_hash import StateHash
from model.state_inference.vae import StateVae
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


class BaseStateAgent(BaseAgent, ABC):

    def __init__(
        self,
        env: gym.Env,
        state_inference_model: StateVae,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(env)
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.optim = self._configure_optimizers(optim_kwargs)

        self.state_hash = StateHash(
            self.state_inference_model.z_dim, self.state_inference_model.z_layers
        )

    def _init_state(self):
        return None

    def _preprocess_obs(self, obs: Tensor) -> Tensor:
        # take in 8bit with shape NxHxWxC
        # convert to float with shape NxCxHxW
        obs = convert_8bit_to_float(obs)
        if obs.ndim == 3:
            return obs.permute(2, 0, 1)
        return obs.permute(0, 3, 1, 2)

    def _get_state_hashkey(self, obs: Tensor):
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs_ = self._preprocess_obs(obs)
        with torch.no_grad():
            z = self.state_inference_model.get_state(obs_)
        return self.state_hash(z)

    @abstractmethod
    def _configure_optimizers(self, optim_kwargs): ...

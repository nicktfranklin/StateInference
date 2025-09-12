from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import FloatTensor, Tensor

from value_iteration.models.vae import StateVae
from value_iteration.task.gridworld import ActType, ObsType
from value_iteration.utils.pytorch_utils import DEVICE, convert_8bit_to_float

from .base_agent import BaseVaeAgent
from .training.rollout_data import BaseBuffer


class RandomVaeAgent(BaseVaeAgent):
    def __init__(self, task, state_inference_model: StateVae, **kwargs) -> None:
        super().__init__(task)
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.n_actions = task.action_space.n

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )

    def _preprocess_obs(self, obs: Tensor) -> Tensor:
        obs = convert_8bit_to_float(obs)
        if obs.ndim == 3:
            return obs.permute(2, 0, 1)
        return obs.permute(0, 3, 1, 2)

    def get_states(self, obs: Tensor):
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs_ = self._preprocess_obs(obs)
        with torch.no_grad():
            z = self.state_inference_model.get_state(obs_)
        return z.dot(self.hash_vector)

    def get_pmf(self, x: FloatTensor) -> np.ndarray:
        batch_size = x.shape[0] if x.ndim > 1 else 1
        uniform_probs = np.ones((batch_size, self.n_actions)) / self.n_actions
        return uniform_probs[0] if batch_size == 1 else uniform_probs

    def get_value_fn(self, x: FloatTensor) -> FloatTensor:
        batch_size = x.shape[0] if x.ndim > 1 else 1
        return torch.zeros(batch_size, 1)

    def predict(
        self,
        obs: ObsType,
        state: Optional[FloatTensor] = None,
        episode_start: Optional[bool] = None,
        deterministic: bool = False,
    ) -> tuple[ActType, Optional[FloatTensor]]:
        action = np.random.randint(0, self.n_actions)
        return action, None

    def update_from_batch(self, batch: BaseBuffer):
        pass

    @property
    def device(self):
        return DEVICE

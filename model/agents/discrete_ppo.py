from typing import Any, Dict, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch import FloatTensor, Tensor

import model.state_inference.vae
from model.agents.base_state_agent import BaseStateAgent
from model.agents.utils.state_hash import StateHash
from model.state_inference.vae import StateVae
from model.training.rollout_data import RolloutDataset
from task.utils import ActType
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


class DiscretePPO(BaseStateAgent):

    def __init__(
        self,
        task,
        state_inference_model: StateVae,
        optim_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(task, state_inference_model, optim_kwargs)

    def get_policy(self, obs: Tensor):
        raise NotImplementedError()

    def get_pmf(self, obs: FloatTensor) -> np.ndarray:
        raise NotImplementedError()

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        raise NotImplementedError()

    def update_from_batch(self, buffer: RolloutDataset, progress_bar: bool = False):
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

import os
import pickle
import random
from typing import Any, Dict, Hashable, Optional

import numpy as np
import torch
from torch import FloatTensor, Tensor, nn
from torch.utils.data import DataLoader, Dataset

import model.state_inference.vae
from model.agents.base_agent import BaseAgent
from model.agents.mdp import (
    TabularRewardEstimator,
    TabularStateActionTransitionEstimator,
    value_iteration,
)
from model.agents.policy import SoftmaxPolicy
from model.data import D4rlDataset, OaroTuple
from model.state_inference.vae import StateVae
from task.utils import ActType
from utils.pytorch_utils import DEVICE, convert_8bit_to_float


class ViDataset(Dataset):
    def __init__(self, dataset: Dict[str, Any]):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["observations"])

    def __getitem__(self, idx):
        return {
            "observations": self.dataset["observations"][idx],
            "next_observations": self.dataset["next_observations"][idx],
            "actions": self.dataset["actions"][idx],
            "rewards": self.dataset["rewards"][idx],
        }


class ValueIterationAgent(BaseAgent):
    TRANSITION_MODEL_CLASS = TabularStateActionTransitionEstimator
    REWARD_MODEL_CLASS = TabularRewardEstimator
    POLICY_CLASS = SoftmaxPolicy
    """
    Value iteration agent. Collects rollouts using Q-learning with an optimistic exploration
    policy on a state-inference model (VAE) and then updates the state-inference model with the
    roll outs. The Value-iteration over the rollouts are used to re-estimate state-action values.

    :param state_inference_model: The VAE used to estimate the State

    """

    def __init__(
        self,
        task,
        state_inference_model: StateVae,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        grad_clip: bool = True,
        batch_size: int = 64,
        gamma: float = 0.99,
        n_iter: int = 1000,
        softmax_gain: float = 1.0,
        epsilon: float = 0.05,
        n_steps: Optional[int] = None,  # None => only update at the end,
        n_epochs: int = 10,
        alpha: float = 0.05,
        dyna_updates: int = 5,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(task)
        self.state_inference_model = state_inference_model.to(DEVICE)

        self.optim = self.state_inference_model.configure_optimizers(optim_kwargs)
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_iter = n_iter
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.alpha = alpha

        self.transition_estimator = self.TRANSITION_MODEL_CLASS()
        self.reward_estimator = self.REWARD_MODEL_CLASS()
        self.policy = self.POLICY_CLASS(
            beta=softmax_gain,
            epsilon=epsilon,
            n_actions=task.action_space.n,
        )

        assert epsilon >= 0 and epsilon < 1.0

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )

        self.num_timesteps = 0
        self.value_function = None
        self.n_dyna_updates = dyna_updates

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
        return z.dot(self.hash_vector)

    def update_rollout_policy(self, rollout_buffer: D4rlDataset) -> None:
        # the rollout policy is a DYNA variant
        # dyna updates (note: this assumes a deterministic enviornment,
        # and this code differes from dyna as we are only using resampled
        # values and not seperately sampling rewards and sucessor states

        # pass the obseration tuple through the state-inference network
        obs, a, r, next_obs = rollout_buffer.get_obs(-1)
        s = self._get_state_hashkey(obs)[0]
        sp = self._get_state_hashkey(next_obs)[0]

        # update the model
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

        # update q-values
        self.update_qvalues(s, a, r, sp)

        # resampling (dyna) updates
        for _ in range(min(len(rollout_buffer), self.n_dyna_updates)):
            # sample observations and actions with replacement
            idx = random.randint(0, len(rollout_buffer) - 1)

            obs, a, _, _ = rollout_buffer.get_obs(idx)

            s = self._get_state_hashkey(obs)[0]

            # draw r, sp from the model
            sp = self.transition_estimator.sample(s, a)
            r = self.reward_estimator.sample(sp)

            self.update_qvalues(s, a, r, sp)

    def get_policy(self, obs: Tensor):
        s = self._get_state_hashkey(obs)
        p = self.policy.get_distribution(s)
        return p

    def get_pmf(self, obs: FloatTensor) -> np.ndarray:
        return self.get_policy(obs).distribution.probs.clone().detach().numpy()

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.random.randint(self.policy.n_actions), None

        s = self._get_state_hashkey(obs)
        p = self.policy.get_distribution(s)
        return p.get_actions(deterministic=deterministic).item(), None

    def update_model(self, obs: OaroTuple):
        s, a, r, sp = self._get_sars_tuples(obs)
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

    def update_qvalues(self, s, a, r, sp) -> None:
        """
        This is the Dyna-Q update rule. From Sutton and Barto (2020), ch 8
        """

        self.policy.maybe_init_q_values(s)
        self.policy.maybe_init_q_values(sp)

        q_s_a = self.policy.q_values[s][a]
        V_sp = max(self.policy.q_values[sp].values())

        q_s_a += self.alpha * (r + self.gamma * V_sp - q_s_a)
        self.policy.q_values[s][a] = q_s_a

    def get_state_values(self) -> torch.Tensor:
        return self.policy.get_value_function()

    def train_vae(self, buffer: D4rlDataset, progress_bar: bool = True):
        # prepare the dataset for training the VAE
        dataset = buffer.get_dataset()
        obs = convert_8bit_to_float(torch.tensor(dataset["observations"])).to(DEVICE)
        obs = obs.permute(0, 3, 1, 2)  # -> NxCxHxW

        dataloader = DataLoader(
            obs, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # train the VAE on the rollouts
        self.state_inference_model.train_epochs(
            self.n_epochs,
            dataloader,
            self.optim,
            self.grad_clip,
            progress_bar=progress_bar,
        )

    def update_from_batch(self, buffer: D4rlDataset, progress_bar: bool = False):
        self.train_vae(buffer, progress_bar=progress_bar)

        # re-estimate the reward and transition functions
        self.reward_estimator.reset()
        self.transition_estimator.reset()

        dataset = buffer.get_dataset()

        # convert ot a tensor dataset for iteration
        dataset = ViDataset(dataset)

        # _get_hashed_state takes care of preprocessing
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        s, sp, a, r = [], [], [], []
        for batch in dataloader:
            s.append(self._get_state_hashkey(batch["observations"]))
            sp.append(self._get_state_hashkey(batch["next_observations"]))
            a.append(batch["actions"])
            r.append(batch["rewards"])
        s = np.concatenate(s)
        sp = np.concatenate(sp)
        a = np.concatenate(a)
        r = np.concatenate(r)

        for idx in range(len(s)):
            self.transition_estimator.update(s[idx], a[idx], sp[idx])
            self.reward_estimator.update(sp[idx], r[idx])

        # use value iteration to estimate the rewards
        self.policy.q_values, value_function = value_iteration(
            t=self.transition_estimator.get_transition_functions(),
            r=self.reward_estimator,
            gamma=self.gamma,
            iterations=self.n_iter,
        )
        self.value_function = value_function

    @classmethod
    def make_from_configs(
        cls,
        task,
        agent_config: Dict[str, Any],
        vae_config: Dict[str, Any],
        env_kwargs: Dict[str, Any],
    ):
        VaeClass = getattr(model.state_inference.vae, agent_config["vae_model_class"])
        vae = VaeClass.make_from_configs(vae_config, env_kwargs)
        return cls(task, vae, **agent_config["state_inference_model"])

    def get_graph_laplacian(self) -> tuple[np.ndarray, Dict[Hashable, int]]:
        return self.transition_estimator.get_graph_laplacian()

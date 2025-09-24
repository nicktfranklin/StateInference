import random
from typing import Any, Dict, Hashable, Optional

import gymnasium as gym
import numpy as np
import torch
from pydantic import BaseModel
from stable_baselines3.common.callbacks import BaseCallback
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from value_iteration.models.vae import StateVae
from value_iteration.task.utils import ActType
from value_iteration.utils.pytorch_utils import DEVICE, convert_8bit_to_float

from .base_agent import BaseVaeAgent
from .training.data import MdpDataset
from .training.rollout_data import BaseBuffer
from .utils.policy import SoftmaxPolicy
from .utils.tabular_agents import (
    TabularRewardEstimator,
    TabularStateActionTransitionEstimator,
    value_iteration,
)


class ValueIterationAgentConfig(BaseModel):
    optim_kwargs: Optional[Dict[str, Any]] = None
    grad_clip: bool | float = 1.0
    batch_size: int = 64
    gamma: float = 0.99
    n_iter: int = 1000
    softmax_gain: float = 1.0
    epsilon: float = 0.05
    n_steps: Optional[int] = None  # None => only update at the end
    n_epochs: int = 10
    alpha: float = 0.05
    dyna_updates: int = 5
    persistant_optim: bool = False
    tensorboard_log: Optional[str] = None  # Path to tensorboard log directory
    log_interval: int = 100  # How often to log metrics


class ValueIterationAgent(BaseVaeAgent):
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
        task: gym.Env,
        vae_model: StateVae,
        config: ValueIterationAgentConfig,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(task)
        self.state_inference_model = vae_model.to(DEVICE)

        self.optim = (
            self.state_inference_model.configure_optimizers(config.optim_kwargs)
            if config.persistant_optim
            else None
        )
        self.grad_clip = config.grad_clip
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.n_iter = config.n_iter
        self.n_steps = config.n_steps
        self.n_epochs = config.n_epochs
        self.alpha = config.alpha

        self.env = task

        self.transition_estimator = self.TRANSITION_MODEL_CLASS()
        self.reward_estimator = self.REWARD_MODEL_CLASS()
        self.policy = self.POLICY_CLASS(
            beta=config.softmax_gain,
            epsilon=config.epsilon,
            n_actions=task.action_space.n,
        )

        assert config.epsilon >= 0 and config.epsilon < 1.0

        self.num_timesteps = 0
        self.value_function = None

        # TensorBoard logging setup
        self.tensorboard_log = config.tensorboard_log
        self.log_interval = config.log_interval
        self.writer = (
            SummaryWriter(log_dir=config.tensorboard_log)
            if config.tensorboard_log
            else None
        )
        self._n_updates = 0

    def _init_state(self):
        return None

    def _preprocess_obs(self, obs: Tensor) -> Tensor:
        # take in 8bit with shape NxHxWxC
        # convert to float with shape NxCxHxW
        obs = convert_8bit_to_float(obs)
        if obs.ndim == 3:
            return obs.permute(2, 0, 1)
        return obs.permute(0, 3, 1, 2)

    def get_states(self, obs: Tensor):
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs_ = self._preprocess_obs(obs)
        with torch.no_grad():
            return self.state_inference_model.get_states(obs_)

    ## Depricated
    # def update_rollout_policy(
    #     self,
    #     obs: int,
    #     a: int,
    #     outcome_tuple,
    #     rollout_buffer: BaseBuffer,
    # ) -> None:
    #     # the rollout policy is a DYNA variant
    #     # dyna updates (note: this assumes a deterministic enviornment,
    #     # and this code differes from dyna as we are only using resampled
    #     # values and not seperately sampling rewards and sucessor states

    #     # pass the obseration tuple through the state-inference network
    #     next_obs, r, _, _, _ = outcome_tuple
    #     s = self.get_states(obs)[0]
    #     sp = self.get_states(next_obs)[0]

    #     # update the model
    #     self.transition_estimator.update(s, a, sp)
    #     self.reward_estimator.update(sp, r)

    #     # update q-values
    #     self.update_qvalues(s, a, r, sp)

    #     # resampling (dyna) updates
    #     for _ in range(min(len(rollout_buffer), self.n_dyna_updates)):
    #         # sample observations and actions with replacement
    #         idx = random.randint(0, len(rollout_buffer) - 1)

    #         obs, a, _, _ = rollout_buffer.get_obs(idx)

    #         s = self.get_states(obs)[0]

    #         # draw r, sp from the model
    #         sp = self.transition_estimator.sample(s, a)
    #         r = self.reward_estimator.sample(sp)

    #         self.update_qvalues(s, a, r, sp)

    def get_policy(self, obs: Tensor):
        s = self.get_states(obs)
        p = self.policy.get_distribution(s)
        return p

    def get_pmf(self, obs: FloatTensor) -> np.ndarray:
        return self.get_policy(obs).distribution.probs.clone().detach().numpy()

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.random.randint(self.policy.n_actions), None

        s = self.get_states(obs)
        p = self.policy.get_distribution(s)
        return p.get_actions(deterministic=deterministic).item(), None

    def update_qvalues(self, s, a, r, sp) -> None:
        """
        This is the Dyna-Q update rule. From Sutton and Barto (2020), ch 8
        """

        self.policy.maybe_init_q_values(s)
        self.policy.maybe_init_q_values(sp)

        old_q_s_a = self.policy.q_values[s][a]
        V_sp = max(self.policy.q_values[sp].values())

        td_error = r + self.gamma * V_sp - old_q_s_a
        new_q_s_a = old_q_s_a + self.alpha * td_error
        self.policy.q_values[s][a] = new_q_s_a

        # Log Q-value update metrics (periodically to avoid too much logging)
        if self.writer is not None and self.num_timesteps % self.log_interval == 0:
            self.writer.add_scalar("q_learning/td_error", td_error, self.num_timesteps)
            self.writer.add_scalar("q_learning/reward", r, self.num_timesteps)

    def get_state_values(self) -> torch.Tensor:
        return self.policy.get_value_function()

    def train_vae(self, buffer: BaseBuffer, progress_bar: bool = True):
        """
        In the Vi Agent, the VAE is trained separately from the policy.
        """

        # prepare the dataset for training the VAE
        dataset = buffer.get_dataset()
        obs = convert_8bit_to_float(torch.tensor(dataset["observations"])).to(DEVICE)
        obs = obs.permute(0, 3, 1, 2)  # -> NxCxHxW

        dataloader = DataLoader(
            obs, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        if self.optim is None:
            optim = self.state_inference_model.configure_optimizers()
        else:
            optim = self.optim

        train_loss, test_loss = self.state_inference_model.fit(
            self.n_epochs,
            train_dataloader=dataloader,
            optim=optim,
            grad_clip=self.grad_clip,
        )

        # Log VAE training metrics
        if self.writer is not None:
            self.writer.add_scalar("vae/train_loss", train_loss[-1], self._n_updates)
            if test_loss is not None:
                self.writer.add_scalar("vae/test_loss", test_loss[-1], self._n_updates)

    def update_from_batch(self, buffer: BaseBuffer, progress_bar: bool = False):
        self.train_vae(buffer, progress_bar=progress_bar)

        # re-estimate the reward and transition functions
        self.reward_estimator.reset()
        self.transition_estimator.reset()

        dataset = buffer.get_dataset()

        # convert ot a tensor dataset for iteration
        dataset = MdpDataset(dataset)

        # _get_hashed_state takes care of preprocessing
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, drop_last=False
        )
        s, sp, a, r = [], [], [], []
        for batch in dataloader:
            s.append(self.get_states(batch["observations"]))
            sp.append(self.get_states(batch["next_observations"]))
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
            T=self.transition_estimator.get_transition_functions(),
            R=self.reward_estimator,
            gamma=self.gamma,
            iterations=self.n_iter,
        )
        self.value_function = value_function

        # Log value iteration metrics
        if self.writer is not None:
            # Log value function statistics
            if value_function is not None:
                self.writer.add_scalar(
                    "value_iteration/mean_value",
                    np.mean(value_function),
                    self._n_updates,
                )
                self.writer.add_scalar(
                    "value_iteration/max_value", np.max(value_function), self._n_updates
                )
                self.writer.add_scalar(
                    "value_iteration/min_value", np.min(value_function), self._n_updates
                )

            # Log Q-values statistics
            if hasattr(self.policy, "q_values") and self.policy.q_values:
                all_q_values = [
                    q
                    for state_q in self.policy.q_values.values()
                    for q in state_q.values()
                ]
                if all_q_values:
                    self.writer.add_scalar(
                        "value_iteration/mean_q_value",
                        np.mean(all_q_values),
                        self._n_updates,
                    )
                    self.writer.add_scalar(
                        "value_iteration/max_q_value",
                        np.max(all_q_values),
                        self._n_updates,
                    )
                    self.writer.add_scalar(
                        "value_iteration/min_q_value",
                        np.min(all_q_values),
                        self._n_updates,
                    )
                    self.writer.add_scalar(
                        "value_iteration/num_states",
                        len(self.policy.q_values),
                        self._n_updates,
                    )

        self._n_updates += 1

    def learn(
        self,
        total_timesteps: int,
        progress_bar: bool = False,
        reset_buffer: bool = False,
        capacity: Optional[int] = None,
        callback: BaseCallback | None = None,
        buffer_class: str | None = None,
        buffer_kwargs: Dict[str, Any] | None = None,
    ):
        # Log hyperparameters at the start of training
        if self.writer is not None:
            self.writer.add_scalar("hyperparameters/gamma", self.gamma, 0)
            self.writer.add_scalar("hyperparameters/alpha", self.alpha, 0)
            self.writer.add_scalar("hyperparameters/epsilon", self.policy.epsilon, 0)
            self.writer.add_scalar("hyperparameters/n_iter", self.n_iter, 0)
            self.writer.add_scalar("hyperparameters/batch_size", self.batch_size, 0)

        result = super().learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            reset_buffer=reset_buffer,
            capacity=capacity,
            callback=callback,
            buffer_class=buffer_class,
            buffer_kwargs=buffer_kwargs,
        )

        # Log total timesteps at the end of training
        if self.writer is not None:
            self.writer.add_scalar(
                "training/total_timesteps", self.num_timesteps, self._n_updates
            )

        return result

    def get_graph_laplacian(
        self, normalized: bool = True
    ) -> tuple[np.ndarray, Dict[Hashable, int]]:
        return self.transition_estimator.get_graph_laplacian(normalized=normalized)

    def close(self):
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __del__(self):
        """Ensure TensorBoard writer is closed when the object is destroyed."""
        self.close()

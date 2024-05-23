from typing import Any, Dict, Hashable, Optional

import gymnasium as gym
import numpy as np
import torch
from torch import FloatTensor, Tensor
from torch.utils.data import DataLoader

import model.state_inference.vae
from model.agents.constants import (
    ALPHA,
    BATCH_SIZE,
    EPSILON,
    GAMMA,
    GRAD_CLIP,
    MAX_SEQUENCE_LEN,
    N_EPOCHS,
    N_ITER_VALUE_ITERATION,
    SOFTMAX_GAIN,
)
from model.agents.utils.mdp import value_iteration
from model.agents.value_iteration import ValueIterationAgent
from model.state_inference.recurrent_vae import LstmVae
from model.training.recurrent import RecurrentDataset
from model.training.rollout_data import OaroTuple, RolloutDataset
from task.utils import ActType
from utils.pytorch_utils import convert_8bit_to_float


class RecurrentViAgent(ValueIterationAgent):
    def __init__(
        self,
        env: gym.Env,
        state_inference_model: LstmVae,
        optim_kwargs: Dict[str, Any] | None = None,
        grad_clip: bool = GRAD_CLIP,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        n_iter: int = N_ITER_VALUE_ITERATION,
        softmax_gain: float = SOFTMAX_GAIN,
        epsilon: float = EPSILON,
        n_steps: Optional[int] = None,  # None => only update at the end,
        n_epochs: int = N_EPOCHS,
        alpha: float = ALPHA,
        dyna_updates: int = 5,
        max_sequence_len: int = MAX_SEQUENCE_LEN,
    ) -> None:
        super().__init__(
            env=env,
            state_inference_model=state_inference_model,
            optim_kwargs=optim_kwargs,
            grad_clip=grad_clip,
            batch_size=batch_size,
            gamma=gamma,
            n_iter=n_iter,
            softmax_gain=softmax_gain,
            epsilon=epsilon,
            n_steps=n_steps,
            n_epochs=n_epochs,
            alpha=alpha,
            dyna_updates=dyna_updates,
        )
        self.max_sequence_len = max_sequence_len

    def _init_state(self):
        return None

    def _preprocess_obs(self, obs: Tensor) -> Tensor:
        """
        obs: (L, B, C, H, W) or (L, C, H, W) or (C, H, W)
        """
        obs = convert_8bit_to_float(obs)
        return obs

    def _get_state_hashkey(self, obs: Tensor):
        """
        obs: (L, B, C, H, W) or (L, C, H, W) or (C, H, W)

        returns:
            (L, B, 1) or (L, 1) or (1)
        """
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)

        obs_ = self._preprocess_obs(obs)

        z = self.state_inference_model.get_state(obs_)
        return z.dot(self.hash_vector)

    def update_rollout_policy(self, rollout_buffer: RolloutDataset) -> None:
        raise NotImplementedError()
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
        raise NotImplementedError()
        s = self._get_state_hashkey(obs)
        p = self.policy.get_distribution(s)
        return p

    def get_pmf(self, obs: FloatTensor) -> np.ndarray:
        raise NotImplementedError()
        return self.get_policy(obs).distribution.probs.clone().detach().numpy()

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        raise NotImplementedError()
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.random.randint(self.policy.n_actions), None

        s = self._get_state_hashkey(obs)
        p = self.policy.get_distribution(s)
        return p.get_actions(deterministic=deterministic).item(), None

    def update_model(self, obs: OaroTuple):
        raise NotImplementedError()
        s, a, r, sp = self._get_sars_tuples(obs)
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

    def update_qvalues(self, s, a, r, sp) -> None:
        raise NotImplementedError()
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
        raise NotImplementedError()
        return self.policy.get_value_function()

    def train_vae(self, dataloader: RecurrentDataset, progress_bar: bool = True):

        # train the VAE on the rollouts
        self.state_inference_model.train_epochs(
            self.n_epochs,
            dataloader,
            self.optim,
            self.grad_clip,
            progress_bar=progress_bar,
        )

    def update_from_batch(self, buffer: RolloutDataset, progress_bar: bool = False):
        # prepare the dataset for training the VAE
        recurrent_dataset = RecurrentDataset(buffer, self.max_sequence_len)

        dataloader = DataLoader(
            recurrent_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=recurrent_dataset.collate_fn,
        )

        self.train_vae(dataloader, progress_bar=progress_bar)

        # re-estimate the reward and transition functions
        self.reward_estimator.reset()
        self.transition_estimator.reset()

        dataset = buffer.get_dataset()

        s, sp, a, r = [], [], [], []
        for batch in recurrent_dataset:

            s_seq = self._get_state_hashkey(batch["obs"])
            print(s_seq)

            # to avoid double counting, we only take a single sars tuple per batch
            s.append(s_seq[-2, 0])
            sp.append(s_seq[-1, 0])
            a.append(batch["action"][-1])
            r.append(batch["reward"][-1])
        s = np.array(s)
        sp = np.array(sp)
        a = np.array(a)
        r = np.array(r)

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

    @classmethod
    def make_from_configs(
        cls,
        task,
        agent_config: Dict[str, Any],
        vae_config: Dict[str, Any],
        env_kwargs: Dict[str, Any],
    ):
        VaeClass = getattr(model.state_inference, agent_config["vae_model_class"])
        vae = VaeClass.make_from_configs(vae_config, env_kwargs)
        return cls(task, vae, **agent_config["state_inference_model"])

    def get_graph_laplacian(self) -> tuple[np.ndarray, Dict[Hashable, int]]:
        return self.transition_estimator.get_graph_laplacian()

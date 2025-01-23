import random
from typing import Any, Dict, Hashable, Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import FloatTensor, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import trange

import src.model.state_inference.vae
from src.model.agents.utils.state_hash import TensorIndexer
from src.task.utils import ActType
from src.utils.pytorch_utils import DEVICE, convert_8bit_to_float

from ..state_inference.vae import StateVae
from ..training.data import MdpDataset, VaeDataset
from ..training.rollout_data import BaseBuffer
from .utils.base_agent import BaseVaeAgent
from .utils.tabular_agent_pytorch import DynaWithViAgent, ModelBasedAgent

# from .utils.tabular_agents import ModelFreeAgent


class LookaheadViAgent(BaseVaeAgent):
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
        grad_clip: int = 2.5,
        batch_size: int = 64,
        gamma: float = 0.99,
        n_iter: int = 1000,
        softmax_gain: float = 1.0,
        epsilon: float = 0.05,
        n_steps: Optional[int] = None,  # None => only update at the end,
        n_epochs: int = 10,
        alpha: float = 0.05,
        dyna_updates: int = 5,
        persistant_optim: bool = False,
    ) -> None:
        """
        :param n_steps: The number of steps to run for each environment per update
        """
        super().__init__(task)
        self.state_inference_model = state_inference_model.to(DEVICE)

        self.optim = (
            self.state_inference_model.configure_optimizers(optim_kwargs)
            if persistant_optim
            else None
        )
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_iter = n_iter
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.epsilon = epsilon

        self.env = task

        n_actions = task.action_space.n

        self.model_based_agent = DynaWithViAgent(
            n_states=0,
            n_actions=n_actions,
            gamma=gamma,
            learning_rate=alpha,
            dyna_updates=dyna_updates,
        )

        z = self.state_inference_model.z_layers * self.state_inference_model.z_dim
        self.actor_net = nn.Sequential(
            nn.Linear(z, z), nn.ReLU(), nn.Linear(z, n_actions), nn.LogSoftmax(dim=1)
        ).to(DEVICE)
        self.critic = nn.Sequential(nn.Linear(z, z), nn.ReLU(), nn.Linear(z, 1)).to(
            DEVICE
        )
        self.ac_optim = torch.optim.Adam(
            list(self.actor_net.parameters()) + list(self.critic.parameters()), lr=alpha
        )

        # self.model_free_agent = ModelFreeAgent(n_actions=n_actions, learning_rate=alpha)
        self.dist = CategoricalDistribution(action_dim=n_actions)
        self.softmax_gain = softmax_gain

        assert epsilon >= 0 and epsilon < 1.0

        self.num_timesteps = 0
        self.value_function = None
        self.n_dyna_updates = dyna_updates

        self.state_indexer = TensorIndexer(device=torch.device("cpu"))

    def _init_state(self):
        return None

    def eval(self):
        self.state_inference_model.eval()

    def actor(self, x):
        x = self.actor_net(x)
        return F.log_softmax(x, dim=1)

    @property
    def device(self):
        return DEVICE

    def reset_state_indexer(self):
        self.state_indexer.reset()

    def _preprocess_obs(self, obs: Tensor) -> Tensor:
        # take in 8bit with shape NxHxWxC
        # convert to float with shape NxCxHxW
        obs = convert_8bit_to_float(obs)
        if obs.ndim == 3:
            return obs.permute(2, 0, 1)
        return obs.permute(0, 3, 1, 2)

    def _get_state_hashkey(self, obs: Tensor, add_to_indexer: bool = False) -> Tensor:
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs_ = self._preprocess_obs(obs)
        with torch.no_grad():
            z = self.state_inference_model.get_state(obs_)
            if add_to_indexer:
                return self.state_indexer.add(z)
            return self.state_indexer(z)

    def update_rollout_policy(
        self,
        obs: int,
        a: int,
        outcome_tuple,
        rollout_buffer: BaseBuffer,
    ) -> None:
        # the rollout policy is a DYNA variant
        # dyna updates (note: this assumes a deterministic enviornment,
        # and this code differes from dyna as we are only using resampled
        # values and not seperately sampling rewards and sucessor states

        # pass the obseration tuple through the state-inference network
        next_obs, r, terimanated, truncated, _ = outcome_tuple
        done = terimanated or truncated

        s = self._get_state_hashkey(obs, add_to_indexer=True).item()
        sp = self._get_state_hashkey(next_obs, add_to_indexer=True).item()

        # # update the model
        self.model_based_agent.update(s, a, r, sp, done)

        # # update q-values
        # self.model_free_agent.update(s, a, r, sp)

        # # resampling (dyna) updates
        # for _ in range(min(len(rollout_buffer), self.n_dyna_updates)):
        #     # sample observations and actions with replacement
        #     idx = random.randint(0, len(rollout_buffer) - 1)

        #     obs, a, _, _ = rollout_buffer.get_obs(idx)

        #     s = self._get_state_hashkey(obs)[0]

        #     # draw r, sp from the model
        #     r, sp = self.model_based_agent.sample(s, a)

        #     self.model_free_agent.update(s, a, r, sp)

    def get_policy(self, obs: Tensor):
        s = self._get_state_hashkey(obs)
        if s == -1:
            return self.dist.proba_distribution(
                torch.ones(self.env.action_space.n) / self.env.action_space.n
            )
        q = self.model_based_agent.get_q_values(s)
        return self.dist.proba_distribution(q * self.softmax_gain)

    def get_pmf(self, obs: FloatTensor) -> np.ndarray:
        if obs.ndim == 3:
            pass
        elif obs.ndim == 4 and obs.shape[0] > 1:
            return np.stack([self.get_pmf(o) for o in obs]).squeeze()

        return self.get_policy(obs).distribution.probs.clone().detach().numpy()

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[ActType, None]:
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_space.n), None

        p = self.get_policy(torch.from_numpy(obs).to(DEVICE))
        return p.get_actions(deterministic=deterministic).item(), None

    def get_state_values(self) -> torch.Tensor:
        return self.model_free_agent.get_value_function()

    def get_critic_values(self, obs: Tensor) -> torch.Tensor:
        self.eval()
        self.state_inference_model.eval()
        with torch.no_grad():
            _obs = self._preprocess_obs(obs).to(DEVICE).float()
            _, z = self.state_inference_model.encode(_obs)
            return self.critic(z.view(1, -1).float()).squeeze()

    def get_actor_values(self, obs: Tensor) -> torch.Tensor:
        self.eval()
        self.state_inference_model.eval()
        with torch.no_grad():
            _obs = self._preprocess_obs(obs).to(DEVICE).float()
            _, z = self.state_inference_model.encode(_obs)
            return self.actor(z.view(1, -1).float()).squeeze()

    def eval(self):
        self.state_inference_model.eval()
        self.actor_net.eval()
        self.critic.eval()

    def train(self):
        self.state_inference_model.train()
        self.actor_net.train()
        self.critic.train()

    def train_vae(self, buffer: BaseBuffer, progress_bar: bool = True):
        # prepare the dataset for training the VAE
        dataset = buffer.get_dataset()
        obs = convert_8bit_to_float(torch.tensor(dataset["observations"])).to(DEVICE)
        next_obs = convert_8bit_to_float(torch.tensor(dataset["next_observations"])).to(
            DEVICE
        )
        obs = obs.permute(0, 3, 1, 2)  # -> NxCxHxW
        next_obs = next_obs.permute(0, 3, 1, 2)  # -> NxCxHxW

        # We use a specific dataset for the VAE training
        dataloader = DataLoader(
            VaeDataset(obs, dataset["actions"], next_obs, rewards=dataset["rewards"]),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        if self.optim is None:
            optim = self.state_inference_model.configure_optimizers()
        else:
            optim = self.optim

        if progress_bar:
            iterator = trange(self.n_epochs, desc="Vae Epochs")
        else:
            iterator = range(self.n_epochs)

        self.train()
        for _ in iterator:
            for batch in dataloader:

                optim.zero_grad()
                loss = self.state_inference_model.loss(
                    batch["obs"], batch["actions"], batch["rewards"], batch["next_obs"]
                )
                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.state_inference_model.parameters(), self.grad_clip
                    )

                optim.step()
            self.state_inference_model.prep_next_batch()
        self.state_inference_model.eval()

    def update_from_batch(self, buffer: BaseBuffer, progress_bar: bool = False):
        self.train_vae(buffer, progress_bar=progress_bar)
        dataloader = self.update_mb_agent(buffer)
        self.update_actor_critic(dataloader)

    def update_mb_agent(self, buffer: BaseBuffer):

        ## prepare the dataset for training the actor-critic
        self.eval()

        dataset = buffer.get_dataset()

        # convert ot a tensor dataset for iteration
        dataset = MdpDataset(dataset)

        # _get_hashed_state takes care of preprocessing
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        # re-estimate the reward and transition functions
        self.reset_state_indexer()

        s, sp, a, r, d = [], [], [], [], []
        for batch in dataloader:
            s.append(
                self._get_state_hashkey(
                    batch["observations"], add_to_indexer=True
                ).item()
            )
            sp.append(
                self._get_state_hashkey(
                    batch["next_observations"], add_to_indexer=True
                ).item()
            )
            a.append(batch["actions"].item())
            r.append(batch["rewards"].item())
            d.append(batch["dones"].item())

        n_states = len(set(s + sp))
        self.model_based_agent.reset(n_states)

        for idx in range(len(s)):
            self.model_based_agent.update(s[idx], a[idx], r[idx], sp[idx], d[idx])

        self.model_based_agent.estimate()
        self.value_function = self.model_based_agent.value_function

        # reverse the state indexer
        states, next_states, values = [], [], []
        for s0, sp0 in zip(s, sp):
            z = (
                F.one_hot(
                    self.state_indexer.lookup(s0), self.state_inference_model.z_dim
                )
                .view(-1)
                .float()
                .to(DEVICE)
            )
            zp = (
                F.one_hot(
                    self.state_indexer.lookup(sp0), self.state_inference_model.z_dim
                )
                .view(-1)
                .float()
                .to(DEVICE)
            )
            states.append(z)
            next_states.append(zp)
            values.append(self.value_function[s0].to(DEVICE))
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        values = torch.stack(values)

        # make a new dataloader
        class MyDataset(torch.utils.data.Dataset):
            def __init__(self, states, next_states, values, rewards, actions):
                self.s = states
                self.sp = next_states
                self.a = actions
                self.r = rewards
                self.v = values

            def __len__(self):
                return len(self.s)

            def __getitem__(self, idx):
                return {
                    "state": self.s[idx],
                    "next_state": self.sp[idx],
                    "action": self.a[idx],
                    "reward": self.r[idx],
                    "values": self.v[idx],
                }

        # print(states, next_states, values, r, a)
        ds = MyDataset(
            states,
            next_states,
            values,
            torch.tensor(r).to(DEVICE),
            torch.tensor(a).to(DEVICE),
        )
        dataloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def update_actor_critic(self, dataloader):
        self.train()
        optim = self.ac_optim
        for _ in range(self.n_epochs):
            for batch in dataloader:

                logits = self.actor_net(batch["state"])
                l2_loss = 0.5 * (logits**2).mean()

                log_probs = F.log_softmax(logits, dim=1)[
                    torch.arange(batch["state"].shape[0]), batch["action"]
                ]
                with torch.no_grad():
                    state_values = self.critic(batch["state"]).view(-1)
                    next_state_values = self.critic(batch["next_state"]).view(-1)

                # compute the advantage
                advantages = (
                    batch["reward"] + self.gamma * next_state_values - state_values
                )
                # normalize the advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # compute the actor loss
                actor_loss = -(log_probs * advantages.detach()).mean()

                # compute the critic loss
                critic_loss = nn.functional.mse_loss(batch["values"], state_values)

                optim.zero_grad()
                loss = actor_loss + critic_loss + l2_loss
                loss.backward()

                optim.step()

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
        super().learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar,
            reset_buffer=reset_buffer,
            capacity=capacity,
            callback=callback,
            buffer_class=buffer_class,
            buffer_kwargs=buffer_kwargs,
        )

    @classmethod
    def make_from_configs(
        cls,
        task,
        agent_config: Dict[str, Any],
        vae_config: Dict[str, Any],
        env_kwargs: Dict[str, Any],
    ):
        VaeClass = getattr(
            src.model.state_inference.vae, agent_config["vae_model_class"]
        )
        vae = VaeClass.make_from_configs(vae_config, env_kwargs)
        return cls(task, vae, **agent_config["state_inference_model"])

    def get_graph_laplacian(
        self, normalized: bool = True
    ) -> tuple[np.ndarray, Dict[Hashable, int]]:
        return self.model_based_agent.get_graph_laplacian(normalized=normalized)

    def get_value_fn(self, batch: BaseBuffer):
        raise NotImplementedError

    def get_states(self, obs: Tensor, add_to_indexer=True) -> Hashable:
        return self._get_state_hashkey(obs, add_to_indexer=add_to_indexer)

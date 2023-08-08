from dataclasses import dataclass
from random import choice
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import CategoricalDistribution
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import trange

from state_inference.gridworld_env import ActType, RewType
from state_inference.model.tabular_models import (
    TabularRewardEstimator,
    TabularStateActionTransitionEstimator,
    value_iteration,
)
from state_inference.model.vae import RecurrentVae, StateVae
from state_inference.utils.data import RecurrentVaeDataset, TransitionVaeDataset
from state_inference.utils.pytorch_utils import (
    DEVICE,
    convert_8bit_to_float,
    maybe_convert_to_tensor,
    train,
)

BATCH_SIZE = 64
N_EPOCHS = 20
MAX_SEQUENCE_LEN = 4
GRAD_CLIP = True
GAMMA = 0.99
N_ITER_VALUE_ITERATION = 1000
SOFTMAX_GAIN = 1.0
EPSILON = 0.05
BATCH_LENGTH = None  # only update at end
ALPHA = 0.05


@dataclass
class OaroTuple:
    obs: Tensor
    a: ActType
    r: RewType
    obsp: Tensor
    index: int  # unique index for each trial


class BaseAgent:
    def __init__(
        self, task, state_inference_model: StateVae, set_action: Set[ActType]
    ) -> None:
        self.task = task
        self.state_inference_model = state_inference_model.to(DEVICE)
        self.set_action = set_action
        self.cached_obs = list()

    def get_env(self):
        ## used for compatibility with stablebaseline code,
        return BaseAlgorithm._wrap_env(self.task, verbose=False, monitor_wrapper=True)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if deterministic:
            return np.array(list(self.set_action)[0]), None
        return choice(list(self.set_action)), None

    def _batch_estimate(self, step: int, last_step: bool) -> None:
        pass

    def _within_batch_update(
        self, obs: OaroTuple, state: Optional[Tensor], state_prev: Optional[Tensor]
    ) -> None:
        pass

    def _init_state(self):
        return None

    def _init_index(self):
        if len(self.cached_obs) == 0:
            return 0
        last_obs = self.cached_obs[-1]
        return last_obs.index + 1

    def learn(
        self,
        total_timesteps: int,
        estimate_batch: bool = True,
        progress_bar: bool = False,
        **kwargs,
    ) -> None:
        # Use the current policy to explore
        obs_prev = self.task.reset()[0]

        episode_start = True
        state_prev = self._init_state()
        idx = self._init_index()

        if progress_bar:
            iterator = trange(total_timesteps)
        else:
            iterator = range(total_timesteps)
        for step in iterator:
            action, state = self.predict(obs_prev, state_prev, episode_start)
            episode_start = False

            obs, rew, terminated, _, _ = self.task.step(action.item())
            assert hasattr(obs, "shape")

            obs_tuple = OaroTuple(
                obs=torch.tensor(obs_prev),
                a=action.item(),
                r=rew,
                obsp=torch.tensor(obs),
                index=idx,
            )

            self._within_batch_update(obs_tuple, state, state_prev)

            self.cached_obs.append(obs_tuple)

            obs_prev = obs
            state_prev = state
            if terminated:
                obs_prev = self.task.reset()[0]
                idx += 1
                assert hasattr(obs, "shape")
                assert not isinstance(obs_prev, tuple)

            if estimate_batch:
                self._batch_estimate(step, step == total_timesteps - 1)


class SoftmaxPolicy:
    def __init__(
        self,
        beta: float,
        epsilon: float,
        n_actions: int = 4,
        q_init: float = 1,
    ):
        self.n_actions = n_actions
        self.q_values = dict()
        self.beta = beta
        self.epsilon = epsilon

        self.dist = CategoricalDistribution(action_dim=n_actions)
        self.q_init = {a: q_init for a in range(self.n_actions)}

    def maybe_init_q_values(self, s: int) -> None:
        if s not in self.q_values:
            if self.q_values:
                q_init = {
                    a: max([max(v.values()) for v in self.q_values.values()])
                    for a in range(self.n_actions)
                }
            else:
                q_init = self.q_init
            self.q_values[s] = q_init

    def get_distribution(self, s: int) -> Tensor:
        def _get_q(s0):
            self.maybe_init_q_values(s0)
            q = torch.tensor(list(self.q_values.get(s0, None).values()))
            return q * self.beta

        q_values = torch.stack([_get_q(s0) for s0 in s])
        return self.dist.proba_distribution(q_values)


class ValueIterationAgent(BaseAgent):
    TRANSITION_MODEL_CLASS = TabularStateActionTransitionEstimator
    REWARD_MODEL_CLASS = TabularRewardEstimator
    POLICY_CLASS = SoftmaxPolicy

    def __init__(
        self,
        task,
        state_inference_model: StateVae,
        set_action: Set[ActType],
        optim_kwargs: Optional[Dict[str, Any]] = None,
        grad_clip: bool = GRAD_CLIP,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        n_iter: int = N_ITER_VALUE_ITERATION,
        softmax_gain: float = SOFTMAX_GAIN,
        epsilon: float = EPSILON,
        batch_length: int = BATCH_LENGTH,
        n_epochs: int = N_EPOCHS,
    ) -> None:
        super().__init__(task, state_inference_model, set_action)

        self.optim = self.state_inference_model.configure_optimizers(optim_kwargs)
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.gamma = gamma
        self.n_iter = n_iter
        self.batch_length = batch_length
        self.n_epochs = n_epochs

        self.transition_estimator = self.TRANSITION_MODEL_CLASS()
        self.reward_estimator = self.REWARD_MODEL_CLASS()
        self.policy = self.POLICY_CLASS(
            beta=softmax_gain,
            epsilon=epsilon,
            n_actions=len(set_action),
        )

        assert epsilon >= 0 and epsilon < 1.0

        self.hash_vector = np.array(
            [
                self.state_inference_model.z_dim**ii
                for ii in range(self.state_inference_model.z_layers)
            ]
        )

    def get_policy(self, obs: Tensor):
        s = self._get_hashed_state(obs)
        p = self.policy.get_distribution(s)
        return p

    def predict(
        self, obs: Tensor, state=None, episode_start=None, deterministic: bool = False
    ) -> tuple[Tensor, None]:
        if not deterministic and np.random.rand() < self.policy.epsilon:
            return np.array([np.random.randint(self.policy.n_actions)]), None

        s = self._get_hashed_state(obs)
        p = self.policy.get_distribution(s)
        return p.get_actions(deterministic=deterministic), None

    def _prep_vae_dataloader(self, batch_size: int):
        r"""
        preps the dataloader for training the State Inference VAE

        Args:
            batch_size (int): The number of samples per batch
        """
        obs = torch.stack([o.obs for o in self.cached_obs])
        obs = convert_8bit_to_float(obs).to(DEVICE)

        return DataLoader(
            obs,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    def _train_vae_batch(self):
        dataloader = self._prep_vae_dataloader(self.batch_size)
        for _ in range(self.n_epochs):
            self.state_inference_model.train()
            train(self.state_inference_model, dataloader, self.optim, self.grad_clip)
            self.state_inference_model.prep_next_batch()

    def _get_hashed_state(self, obs: Tensor):
        obs = obs if isinstance(obs, Tensor) else torch.tensor(obs)
        obs_ = convert_8bit_to_float(obs)
        z = self.state_inference_model.get_state(obs_)
        return z.dot(self.hash_vector)

    def _get_sars_tuples(self, obs: OaroTuple):
        s = self._get_hashed_state(obs.obs)[0]
        sp = self._get_hashed_state(obs.obsp)[0]
        return s, obs.a, obs.r, sp

    def update_model(self, obs: OaroTuple):
        s, a, r, sp = self._get_sars_tuples(obs)
        self.transition_estimator.update(s, a, sp)
        self.reward_estimator.update(sp, r)

    def _precalculate_states_for_batch_training(self) -> Tuple[Tensor, Tensor]:
        # pass all of the observations throught the model (really, twice)
        # for speed. It's much faster in a batch than single observations
        obs = torch.stack([o.obs for o in self.cached_obs])
        obsp = torch.stack([o.obsp for o in self.cached_obs])
        s = self._get_hashed_state(obs)
        sp = self._get_hashed_state(obsp)
        return s, sp

    def retrain_model(self):
        self.transition_estimator.reset()
        self.reward_estimator.reset()

        s, sp = self._precalculate_states_for_batch_training()

        for idx, obs in enumerate(self.cached_obs):
            self.transition_estimator.update(s[idx], obs.a, sp[idx])
            self.reward_estimator.update(sp[idx], obs.r)

    def _batch_estimate(self, step: int, last_step: bool) -> None:
        # update only periodically
        if (not last_step) and (
            self.batch_length is None or step == 0 or step % self.batch_length != 0
        ):
            return

        # update the state model
        self._train_vae_batch()

        # construct a devono model-based learner from the new states
        self.retrain_model()

        # use value iteration to estimate the rewards
        self.policy.q_values, value_function = value_iteration(
            t=self.transition_estimator.get_transition_functions(),
            r=self.reward_estimator,
            gamma=self.gamma,
            iterations=self.n_iter,
        )
        self.value_function = value_function


class ViAgentWithExploration(ValueIterationAgent):
    def __init__(
        self,
        task,
        state_inference_model: StateVae,
        set_action: Set[ActType],
        optim_kwargs: Dict[str, Any] | None = None,
        grad_clip: bool = GRAD_CLIP,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        n_iter: int = N_ITER_VALUE_ITERATION,
        softmax_gain: float = SOFTMAX_GAIN,
        epsilon: float = EPSILON,
        batch_length: int = BATCH_LENGTH,
        n_epochs: int = N_EPOCHS,
        alpha: float = ALPHA,
    ) -> None:
        super().__init__(
            task,
            state_inference_model,
            set_action,
            optim_kwargs,
            grad_clip,
            batch_size,
            gamma,
            n_iter,
            softmax_gain,
            epsilon,
            batch_length,
            n_epochs,
        )
        self.alpha = alpha

    def update_qvalues(self, s, a, r, sp) -> None:
        self.policy.maybe_init_q_values(s)
        q_s_a = self.policy.q_values[s][a]

        self.policy.maybe_init_q_values(sp)
        V_sp = max(self.policy.q_values[sp].values())

        q_s_a = (1 - self.alpha) * q_s_a + self.alpha * (r + self.gamma * V_sp)
        self.policy.q_values[s][a] = q_s_a

    def _within_batch_update(
        self, obs: OaroTuple, state: Optional[Tensor], state_prev: Optional[Tensor]
    ) -> None:
        # state and state_prev are only used by recurrent models
        assert state is None
        assert state_prev is None

        s, a, r, sp = self._get_sars_tuples(obs)
        self.update_qvalues(s, a, r, sp)


class ViControlableStateInf(ViAgentWithExploration):
    def _prep_vae_dataloader(self, batch_size: int = BATCH_SIZE):
        obs = torch.stack([obs.obs for obs in self.cached_obs])
        obsp = torch.stack([obs.obsp for obs in self.cached_obs])
        a = F.one_hot(
            torch.tensor([obs.a for obs in self.cached_obs]), num_classes=4
        ).float()

        dataset = TransitionVaeDataset(obs, a, obsp)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )


class ViDynaAgent(ViControlableStateInf):
    k = 10

    def _within_batch_update(self, obs: OaroTuple) -> None:
        # update the current q-value
        s, a, r, sp = self._get_sars_tuples(obs)
        self.update_qvalues(s, a, r, sp)

        # dyna updates (note: this assumes a deterministic enviornment,
        # and this code differes from dyna as we are only using resampled
        # values and not seperately sampling rewards and sucessor states
        if self.cached_obs:
            for _ in range(self.k):
                obs = choice(self.cached_obs)
                s, a, r, sp = self._get_sars_tuples(obs)
                self.update_qvalues(s, a, r, sp)


class RecurrentStateInf(ViAgentWithExploration):
    def __init__(
        self,
        task,
        state_inference_model: RecurrentVae,
        set_action: Set[ActType],
        optim_kwargs: Dict[str, Any] | None = None,
        grad_clip: bool = GRAD_CLIP,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        n_iter: int = N_ITER_VALUE_ITERATION,
        softmax_gain: float = SOFTMAX_GAIN,
        epsilon: float = EPSILON,
        batch_length: int = BATCH_LENGTH,
        n_epochs: int = N_EPOCHS,
        alpha: float = ALPHA,
        max_sequence_len: int = MAX_SEQUENCE_LEN,
    ) -> None:
        super().__init__(
            task,
            state_inference_model,
            set_action,
            optim_kwargs,
            grad_clip,
            batch_size,
            gamma,
            n_iter,
            softmax_gain,
            epsilon,
            batch_length,
            n_epochs,
            alpha,
        )
        self.max_sequence_len = max_sequence_len

    @staticmethod
    def construct_dataloader_from_obs(
        batch_size: int, list_obs: List[Tensor], max_seq_len: int = MAX_SEQUENCE_LEN
    ) -> DataLoader:
        obs = torch.stack([obs.obs for obs in list_obs])
        actions = F.one_hot(torch.tensor([obs.a for obs in list_obs]))
        index = [obs.index for obs in list_obs]
        return RecurrentVaeDataset.construct_dataloader(
            obs, actions, index, max_seq_len, batch_size
        )

    def _prep_vae_dataloader(
        self,
        batch_size: int,
    ):
        r"""
        preps the dataloader for training the State Inference VAE

        Args:
            batch_size (int): The number of samples per batch
        """
        return RecurrentStateInf.construct_dataloader_from_obs(
            batch_size, self.cached_obs, self.max_sequence_len
        )

    def contruct_validation_dataloader(self, sample_size, seq_len):
        # assert (
        #     sample_size % seq_len == 0
        # ), "Sample size must be an interger multiple of sequence length"
        validation_obs = []
        for t in range(sample_size // seq_len):
            obs_prev = self.task.reset()[0]

            for _ in range(seq_len):
                action = choice(list(self.set_action))
                obs, rew, terminated, _, _ = self.task.step(action)
                obs_tuple = OaroTuple(
                    obs=torch.tensor(obs_prev),
                    a=action,
                    r=rew,
                    obsp=torch.tensor(obs),
                    index=t,
                )
                validation_obs.append(obs_tuple)

                if terminated:
                    break
        return RecurrentStateInf.construct_dataloader_from_obs(
            batch_size=len(validation_obs), list_obs=validation_obs, max_seq_len=seq_len
        )

    def _precalculate_states_for_batch_training(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _init_state(self) -> Tensor:
        # unbatch state for the agent (not vae training)
        return torch.zeros(
            self.state_inference_model.z_dim * self.state_inference_model.z_layers
        )

    def _get_hashed_state(self, obs: Tensor, state_prev: Optional[Tensor]):
        obs = maybe_convert_to_tensor(obs)
        obs = convert_8bit_to_float(obs)
        state = self.state_inference_model.get_state(obs, state_prev)
        return state.dot(self.hash_vector)

    def predict(
        self,
        obs: Tensor,
        state: Tensor,
        episode_start=None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor]:
        obs = maybe_convert_to_tensor(obs)
        obs = convert_8bit_to_float(obs)

        # e-greedy sampling
        if not deterministic and np.random.rand() < self.policy.epsilon:
            a = torch.tensor([np.random.randint(self.policy.n_actions)])
        else:
            hashed_sucessor_state = self._get_hashed_state(obs, state)

            p = self.policy.get_distribution(hashed_sucessor_state)

            # sample the action
            a = p.get_actions(deterministic=deterministic)

        # update the RNN Hidden state
        next_state = self.state_inference_model.update_hidden_state(obs, state, a)

        return a, next_state

    def _within_batch_update(
        self, obs: OaroTuple, state: None, state_prev: None
    ) -> None:
        # prep the observations for the RNN
        o = convert_8bit_to_float(obs.obs.view(1, -1)).to(DEVICE)
        op = convert_8bit_to_float(obs.obsp.view(1, -1)).to(DEVICE)

        state = state.view(1, -1).to(DEVICE)
        state_prev = state_prev.view(1, -1).to(DEVICE)

        # pass through the RNN to get the embedding
        _, z = self.state_inference_model.encode_from_state(o, state)
        _, zp = self.state_inference_model.encode_from_state(op, state_prev)

        # convert from one hot tensor to int array
        s = torch.argmax(z, dim=-1).detach().cpu().view(-1).numpy()
        sp = torch.argmax(zp, dim=-1).detach().cpu().view(-1).numpy()

        s = s.dot(self.hash_vector)
        sp = sp.dot(self.hash_vector)
        self.update_qvalues(s, obs.a, obs.r, sp)

from collections import Counter, namedtuple
from dataclasses import dataclass
from random import choice, choices
from typing import Dict, Hashable, List, Optional, Set, Union

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from sklearn.neighbors import KNeighborsClassifier

from model.vae import StateVae
from task.gridworld import ActType, GridWorldEnv, ObsType
from task.observation_model import ObservationModel
from utils.pytorch_utils import DEVICE, convert_8bit_array_to_float_tensor


class RandomAgent:
    def __init__(
        self, state_inference_model: StateVae, set_action: Set[Union[int, str]]
    ) -> None:
        self.state_inference_model = state_inference_model
        self.set_action = set_action

    def get_hashed_state(self, obs: int) -> tuple[int]:
        return tuple(*self.state_inference_model.get_state(obs))

    def predict(self, observation: ObsType) -> ActType:
        return choice(list(self.set_action))

    def update(self, o_prev: ObsType, o: ObsType, a: ActType, r: float) -> None:
        pass


class OnPolicyCritic(RandomAgent):
    def __init__(
        self,
        state_inference_model: StateVae,
        set_action: Set[Union[int, str]],
        gamma: float = 0.9,
        n_iter=1000,
    ) -> None:
        super().__init__(state_inference_model, set_action)
        self.transitions = TabularTransitionEstimator()
        self.rewards = TabularRewardEstimator()
        self.gamma = gamma
        self.n_iter = n_iter
        self.values = None

    def update(self, o_prev: ObsType, o: ObsType, a: ActType, r: float) -> None:
        s_prev, s = self.get_hashed_state(o_prev), self.get_hashed_state(o)

        self.rewards.update(s_prev, r)
        self.transitions.update(s_prev, s)

    def sample_policy(self, observation: ObsType) -> Union[str, int]:
        """For debugging, has a random policy"""
        return choice(list(self.set_action))

    def update_values(self):
        _, self.values = value_iteration(
            {"N": self.transitions},
            self.rewards,
            gamma=self.gamma,
            iterations=self.n_iter,
        )


class Sarsa(RandomAgent):
    default_q_value = 0.001  # optimistic

    def __init__(
        self,
        state_inference_model: StateVae,
        set_action: Set[Union[int, str]],
        learning_rate: float = 0.1,
        gamma: float = 0.80,
    ) -> None:
        super().__init__(state_inference_model, set_action)

        self.alpha = learning_rate
        self.gamma = gamma

        self.q_values = dict()
        self.default_q_function = {a: self.default_q_value for a in set_action}

        # # keep track of previous actions for on-policy updates
        self.a_next = self._pre_sample_policy(0)  # initialize with a random action

    def update(self, o_prev: ObsType, o: ObsType, a: ActType, r: float) -> None:
        s_prev, s = self.get_hashed_state(o_prev), self.get_hashed_state(o)

        # pre-choose the next action acording to current policy
        self.a_next = self._pre_sample_policy(s)

        # update the q values with the on-policy algorithm
        q = self.q_values.get(s_prev, self.default_q_function)
        qp = self.q_values.get(s, self.default_q_function)
        pe = r + self.gamma + qp[self.a_next] - q[a]

        q[a] = q[a] + self.alpha * pe
        self.q_values[s_prev] = q

    def _pre_sample_policy(self, s: Hashable) -> Union[str, int]:
        q = self.q_values.get(s, self.default_q_function)
        z = logsumexp(list(q.values()))
        p = {action: np.exp(v - z) for action, v in q.items()}
        return choices(list(p.keys()), weights=list(p.values()))

    def predict(self, observation: Hashable) -> Union[str, int]:
        return self.a_next

    def sample_policy(self, observation: ObsType) -> Union[str, int]:
        """For debugging, has a random policy"""
        return choice(list(self.set_action))


class TabularTransitionEstimator:
    ## Note: does not take in actions

    def __init__(self, *args, **kwargs):
        self.transitions = dict()
        self.pmf = dict()

    def reset(self):
        self.transitions = dict()
        self.pmf = dict()

    def update(self, s: Hashable, sp: Hashable):
        if s in self.transitions:
            if sp in self.transitions[s]:
                self.transitions[s][sp] += 1
            else:
                self.transitions[s][sp] = 1
        else:
            self.transitions[s] = {sp: 1}

        N = float(sum(self.transitions[s].values()))
        self.pmf[s] = {sp0: v / N for sp0, v in self.transitions[s].items()}

    def batch_update(self, list_states: List[Hashable]):
        for ii in range(len(list_states) - 1):
            self.update(list_states[ii], list_states[ii + 1])

    def get_transition_probs(self, s: Hashable) -> Dict[Hashable, float]:
        # if a state is not in the model, assume it's self-absorbing
        if s not in self.pmf:
            return {s: 1.0}
        return self.pmf[s]


class TabularStateActionTransitionEstimator:
    def __init__(self, n_actions: int = 4):
        self.n_actions = n_actions
        self.models = {a: TabularTransitionEstimator() for a in range(n_actions)}
        self.set_states = set()

    def reset(self):
        for m in self.models.values():
            m.reset()
        self.set_states = set()

    def update(self, s: Hashable, a: ActType, sp: Hashable) -> None:
        self.models[a].update(s, sp)
        self.set_states.add(s)

    def get_transition_functions(self):
        return self.models


class TabularRewardEstimator:
    def __init__(self):
        self.counts = dict()
        self.state_reward_function = dict()

    def reset(self):
        self.counts = dict()
        self.state_reward_function = dict()

    def update(self, s: Hashable, r: float):
        if s in self.counts.keys():  # pylint: disable=consider-iterating-dictionary
            self.counts[s] += np.array([float(r), 1.0])
        else:
            self.counts[s] = np.array([float(r), 1.0])

        self.state_reward_function[s] = self.counts[s][0] / self.counts[s][1]

    def batch_update(self, s: List[Hashable], r: List[float]):
        for s0, r0 in zip(s, r):
            self.update(s0, r0)

    def get_states(self):
        return list(self.state_reward_function.keys())

    def get_reward(self, state):
        return self.state_reward_function.get(state, np.nan)


def value_iteration(
    t: Dict[ActType, TabularTransitionEstimator],
    r: TabularRewardEstimator,
    gamma: float,
    iterations: int,
):
    list_states = r.get_states()
    list_actions = list(t.keys())
    q_values = {s: {a: 0 for a in list_actions} for s in list_states}
    v = {s: 0 for s in list_states}

    def _successor_value(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * v[sp]
        return _sum

    def _expected_reward(s, a):
        _sum = 0
        for sp, p in t[a].get_transition_probs(s).items():
            _sum += p * r.get_reward(sp)
        return _sum

    for _ in range(iterations):
        for s in list_states:
            for a in list_actions:
                q_values[s][a] = _expected_reward(s, a) + gamma * _successor_value(s, a)
        # update value function
        for s, qs in q_values.items():
            v[s] = max(qs.values())

    return q_values, v


SarsTuple = namedtuple("SarsTuple", ["s", "a", "sp", "r"])


@dataclass
class Step:
    s: int
    sp: int
    r: float
    o_prev: torch.Tensor
    o: torch.Tensor
    a: int


@dataclass
class TrialResults:
    s: List[int]
    sp: List[int]
    r: List[float]
    o_prev: List[torch.Tensor]
    o: List[torch.Tensor]
    a: List[int]

    @classmethod
    def make(cls, results: List[Step]):
        s, sp, r, o, o_prev, a = [], [], [], [], [], []
        for t in results:
            s.append(t.s)
            sp.append(t.sp)
            r.append(t.r)
            o.append(t.o)
            o_prev.append(t.o_prev)
            a.append(t.a)
        return TrialResults(s, sp, r, o_prev, o, a)

    def get_total_reward(self):
        return np.sum(self.r)

    def get_visitation_history(self, h: int, w: int) -> np.ndarray:
        history = np.zeros(h * w)
        for s, c in Counter(self.sp).items():
            history[s] = c
        return history.reshape(h, w)

    def plot_cumulative_reward(
        self, ax: Optional[matplotlib.axes.Axes] = None, label: Optional[str] = None
    ) -> None:
        if ax == None:
            plt.figure()
            ax = plt.gca()
        ax.plot(np.cumsum(self.r), label=label)


class Simulator:
    def __init__(
        self, task: GridWorldEnv, agent: RandomAgent, max_trial_length: int = 100
    ) -> None:
        self.task = task
        self.agent = agent
        self.max_trial_length = max_trial_length
        self.obs_prev = None

    def _check_end(self, t: int, is_terminated: bool) -> bool:
        if is_terminated or self.max_trial_length is None:
            return True
        return self.max_trial_length > t

    def initialize_observation(self, o: ObsType) -> None:
        self.obs_prev = o

    @staticmethod
    def preprocess_obs(o: ObsType) -> torch.FloatTensor:
        return convert_8bit_array_to_float_tensor(o)

    def task_reset(self):
        return self.preprocess_obs(self.task.reset()[0])

    def task_step(self, a: ActType):
        o, r, is_terminated, _, _ = self.task.step(a)
        return self.preprocess_obs(o), r, is_terminated, _, _

    def task_observation(self):
        return self.preprocess_obs(self.task.reset()[0])

    def simulate_trial(self) -> TrialResults:
        self.obs_prev = self.task_reset()

        o_prev = self.task_observation()
        s = self.task.get_state()
        t = 0
        trial_history = []
        is_terminated = False

        while self._check_end(t, is_terminated):
            a = self.agent.sample_policy(o_prev)
            o, r, is_terminated, _, _ = self.task_step(a)

            self.agent.update(o_prev, o, a, r)

            trial_history.append(
                Step(s=s, a=a, sp=self.task.get_state(), r=r, o_prev=o_prev, o=o)
            )
            s = self.task.get_state()

            t += 1
        return TrialResults.make(trial_history)


class StateReconstruction:
    def __init__(
        self,
        vae_model: StateVae,
        observation_model: ObservationModel,
        train_states: List[int],
    ):
        n = len(train_states)
        train_observations = torch.stack(
            convert_8bit_array_to_float_tensor(observation_model(train_states))
        ).view(n, -1)

        # encodes the states
        z_train = vae_model.get_state(train_observations.to(DEVICE))

        self.clf = KNeighborsClassifier(
            n_neighbors=100, weights="distance", metric="cityblock"
        ).fit(z_train, train_states)

        self.observation_model = observation_model
        self.vae_model = vae_model

    def _embed(self, states: List[int]):
        n = len(states)
        obs_test = torch.stack(
            convert_8bit_array_to_float_tensor(self.observation_model(states))
        ).view(n, -1)
        embed_state_vars = self.vae_model.get_state(obs_test.to(DEVICE))
        return embed_state_vars

    def predict_prob(self, states: List[int]):
        return self.clf.predict_proba(self._embed(states))

    def predict_state_prob_from_z(self, state_vars: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(state_vars)

    @staticmethod
    def log_loss_by_time(pred_log_probs: np.ndarray, states: List[int]):
        return np.array([ps[s] for ps, s in zip(*[pred_log_probs, states])])

    def cross_entropy(self, pred_log_probs: np.ndarray, states: List[int]):
        return -self.log_loss_by_time(pred_log_probs, states).mean()

    def accuracy(self, pred_log_probs: np.ndarray, states: List[int]):
        return np.exp(self.log_loss_by_time(pred_log_probs, states))

    @staticmethod
    def entropy(log_probability: np.ndarray):
        return -np.sum([log_probability * np.exp(log_probability)], axis=1)

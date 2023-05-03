from abc import ABC, abstractmethod
from collections import Counter, namedtuple
from dataclasses import dataclass
from random import choice
from typing import Dict, Hashable, List, Set, Union

import numpy as np
import torch
from torch import Tensor

from state_inference.env import OaorTuple, WorldModel
from state_inference.model import StateVae


class TabularAgent(ABC):
    def __init__(
        self, state_inference_model: StateVae, set_action: Set[Union[int, str]]
    ) -> None:
        self.state_inference_model = state_inference_model
        self.set_action = set_action

    def get_state(self, obs: int) -> Tensor:
        return self.state_inference_model.get_state(obs)

    @abstractmethod
    def sample_policy(
        self, observation: Union[torch.Tensor, np.ndarray]
    ) -> Union[str, int]:
        ...

    @abstractmethod
    def update(self, obs_tuple: OaorTuple) -> None:
        ...


class OnPolicyCritic(TabularAgent):
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

    def sample_policy(self, observation: Union[Tensor, np.ndarray]) -> Union[str, int]:
        """For debugging, has a random policy"""
        return choice(list(self.set_action))

    def update(self, obs_tuple: OaorTuple) -> None:
        o, _, op, r = obs_tuple
        s, sp = self.get_state(o), self.get_state(op)

        self.rewards.update(sp, r)
        self.transitions.update(s, sp)

    def update_values(self):
        _, self.values = value_iteration(
            {"N": self.transitions},
            self.rewards,
            gamma=self.gamma,
            iterations=self.n_iter,
        )


class TabularTransitionEstimator:
    ## Note: does not take in actions

    def __init__(self):
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


class TabularRewardEstimator:
    def __init__(self):
        self.counts = dict()
        self.state_reward_function = dict()

    def update(self, s: Hashable, r: float):
        if s in self.counts.keys():  # pylint: disable=consider-iterating-dictionary
            self.counts[s] += np.array([r, 1])
        else:
            self.counts[s] = np.array([r, 1])

        self.state_reward_function[s] = self.counts[s][0] / self.counts[s][1]

    def batch_update(self, s: List[Hashable], r: List[float]):
        for s0, r0 in zip(s, r):
            self.update(s0, r0)

    def get_states(self):
        return list(self.state_reward_function.keys())

    def get_reward(self, state):
        return self.state_reward_function[state]


def value_iteration(
    t: Dict[Union[str, int], TabularTransitionEstimator],
    r: TabularRewardEstimator,
    gamma: float,
    iterations: int,
):
    list_states = r.get_states()
    list_actions = list(t.keys())
    q_values = {s: {a: 0 for a in list_actions} for s in list_states}
    v = {s: 0 for s in list_states}

    def _inner_sum(s, a):
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
                q_values[s][a] = _expected_reward(s, a) + gamma * _inner_sum(s, a)
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
    o: Tensor
    op: Tensor
    a: int


@dataclass
class TrialResults:
    s: List[int]
    sp: List[int]
    r: List[float]
    o: List[Tensor]
    op: List[Tensor]
    a: List[int]

    @classmethod
    def make(cls, results: List[Step]):
        s, sp, r, o, op, a = [], [], [], [], [], []
        for t in results:
            s.append(t.s)
            sp.append(t.sp)
            r.append(t.r)
            o.append(t.o)
            op.append(t.op)
            a.append(t.a)
        return TrialResults(s, sp, r, o, op, a)

    def get_total_reward(self):
        return np.sum(self.r)

    def get_visitation_history(self, h: int, w: int) -> np.ndarray:
        history = np.zeros(h * w)
        for s, c in Counter(self.sp).items():
            history[s] = c
        return history.reshape(h, w)


class Simulator:
    def __init__(
        self, task: WorldModel, agent: TabularAgent, max_trial_length: int = 100
    ) -> None:
        self.task = task
        self.agent = agent
        self.max_trial_length = max_trial_length

    def _check_end(self, t: int) -> bool:
        if self.max_trial_length is None:
            return True
        return self.max_trial_length > t

    def simulate_trial(self) -> TrialResults:
        self.task.reset_trial()

        o = self.task.get_obseservation()
        s = self.task.get_state()
        t = 0
        trial_history = []
        while self._check_end(t):
            a = self.agent.sample_policy(o)
            oaor = self.task.take_action(a)

            self.agent.update(oaor)

            trial_history.append(
                Step(s=s, a=a, sp=self.task.get_state(), r=oaor.r, o=oaor.o, op=oaor.op)
            )
            s = self.task.get_state()

            t += 1
        return TrialResults.make(trial_history)
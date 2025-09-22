from dataclasses import dataclass
from random import choices
from typing import Any, Dict, Hashable, List, Optional

import numpy as np

from value_iteration.task.utils import ActType
from value_iteration.utils.sampling_functions import inverse_cmf_sampler


class TabularTransitionEstimator:
    ## Note: does not take in actions

    def __init__(
        self,
        transitions: Optional[Dict[str, Any]] = None,
        pmf: Optional[Dict[str, Any]] = None,
        terminal_state: str = "terminal",
    ):
        self.transitions = (
            transitions if transitions else {terminal_state: {terminal_state: 1}}
        )
        self.pmf = pmf if pmf else {}
        self.terminal_state = terminal_state

    def reset(self):
        self.transitions: Dict[Hashable, int] = {}
        self.pmf: Dict[Hashable, float] = {}

    def update(self, s: Hashable, sp: Hashable, done: bool = False):
        """Updates the transition model with a new transition from state s to state sp"""

        # keep track of the number of transitions from s to sp
        if s in self.transitions:
            if sp in self.transitions[s]:
                self.transitions[s][sp] += 1
            else:
                self.transitions[s][sp] = 1
        else:
            self.transitions[s] = {sp: 1}

        if done:
            # if we are done, we assume the next state is the terminal state
            if sp not in self.transitions.keys():
                self.transitions[sp] = {self.terminal_state: 1}
            elif self.terminal_state in self.transitions[sp]:
                self.transitions[sp][self.terminal_state] += 1
            else:
                self.transitions[sp][self.terminal_state] = 1

        # update the pmf via MLE
        N = float(sum(self.transitions[s].values()))
        self.pmf[s] = {sp0: v / N for sp0, v in self.transitions[s].items()}

    def batch_update(self, list_states: List[Hashable]):
        for ii in range(len(list_states) - 1):
            self.update(list_states[ii], list_states[ii + 1])

    def get_transition_probs(self, s: Hashable) -> Dict[Hashable, float]:
        """Returns the transition probabilities from state s. Assume the state transitions deterministically
        to a self-absorbing state if not in the model or if there is no data for it."""

        # if a state is not in the model, assume it's transitions to a self-absorbing state
        if s not in self.pmf:
            return {self.terminal_state: 1.0}
        return self.pmf[s]

    def __call__(self, s: Hashable) -> Dict[Hashable, float]:
        return self.get_transition_probs(s)

    def sample(self, state: Hashable) -> Hashable:
        pmf = self.get_transition_probs(state)

        sucessor_states = list(pmf.keys())
        pmf = np.array(list(pmf.values()))

        idx = inverse_cmf_sampler(pmf)

        return sucessor_states[idx]


class TabularStateActionTransitionEstimator:

    def __init__(
        self,
        n_actions: int = 4,
        models: Optional[Dict[str, TabularTransitionEstimator]] = None,
        states: Optional[set] = None,
        terminal_state: str = "terminal",
    ):
        self.n_actions = n_actions
        if models is not None:
            self.models = {
                a: TabularTransitionEstimator(**v) for a, v in models.items()
            }
        else:
            self.models = {
                a: TabularTransitionEstimator(terminal_state=terminal_state)
                for a in range(n_actions)
            }
        self.set_states = states if states else set([terminal_state])
        self.terminal_state = terminal_state

    def reset(self):
        for m in self.models.values():
            m.reset()
        self.set_states = set()

    def update(self, s: Hashable, a: ActType, sp: Hashable, done: bool) -> None:
        self.models[a].update(s, sp, done)
        self.set_states.add(s)
        self.set_states.add(sp)

    def get_states(self):
        return list(self.data.keys())

    def get_transition_functions(self):
        return self.models

    def sample(self, s: Hashable, a: ActType) -> None:
        return self.models[a].sample(s)

    def __call__(self, s: Hashable, a: ActType) -> Dict[Hashable, float]:
        return self.models[a].get_transition_probs(s)

    def get_graph_laplacian(
        self, normalized: bool = True, terminal_state: str | None = None
    ) -> tuple[np.ndarray, Dict[Hashable, int]]:
        # the graph is equivalent to a permutation of the states, so
        # we can just pick an arbitrary order for them.
        state_key = {s: ii for ii, s in enumerate(self.set_states)}
        print(f"Found {len(self.set_states)} states")

        adjacency = np.zeros((len(self.set_states), len(self.set_states)))

        for state in self.set_states:
            for action in range(self.n_actions):
                for sp, p in self.models[action].get_transition_probs(state).items():
                    if sp == self.terminal_state:
                        continue
                    assert sp in state_key, f"state {sp} not in state_key"
                    adjacency[state_key[state], state_key[sp]] = 1

        degree_matrix = np.diag(adjacency.sum(axis=1))

        if normalized:
            laplacian = (degree_matrix**0.5) @ adjacency @ (degree_matrix**0.5)
        else:
            laplacian = degree_matrix - adjacency

        return laplacian, state_key


class ActionRewardEstimator:
    """
    A simple class to estimate the value of an action based on the rewards. Uses an append-only log and a MLE
    """

    def __init__(self, n_actions: int = 4, default_value: float = 0.0):
        self.history = {a: [] for a in range(n_actions)}
        self.default_value = default_value

    def reset(self):
        self.history = {a: [] for a in self.counts.keys()}

    def __call__(self, a: int):
        if self.history[a]:
            return np.mean(self.history[a])
        return self.default_value

    def update(self, a: int, r: float):
        self.history[a].append(r)

    def sample(self, a: int):
        if self.history[a]:
            return np.random.choice(self.history[a])
        return self.default_value


class TabularRewardEstimator:

    def __init__(
        self,
        n_actions: int = 4,
        terminal_state: str = "terminal",
        default_reward: float = 0.0,
    ):
        self.n_actions = n_actions
        self.terminal_state = terminal_state
        self.default_reward = default_reward
        self.reset()

    def reset(self):
        self.data = {
            self.terminal_state: ActionRewardEstimator(
                self.n_actions, self.default_reward
            )
        }

    def update(self, s: Hashable, a: ActType, r: float):
        if s not in self.data.keys():
            self.data[s] = ActionRewardEstimator(self.n_actions, self.default_reward)
        self.data[s].update(a, r)

    def batch_update(self, s: List[Hashable], r: List[float]):
        for s0, r0 in zip(s, r):
            self.update(s0, r0)

    def get_states(self):
        return list(self.data.keys())

    def __call__(self, s: Hashable, a: ActType):
        if s is self.terminal_state:
            return 0.0
        return self.data[s](a)

    def sample(self, s: Hashable, a: ActType | None = None):
        if a is None:
            a = np.random.randint(self.n_actions)

        if s not in self.data.keys():
            return self.default_reward

        return self.data[s].sample(a)


def value_iteration(
    T: Dict[ActType, TabularTransitionEstimator],
    R: TabularRewardEstimator,
    gamma: float,
    iterations: int,
) -> tuple[Dict[Hashable, Dict[ActType, float]], Dict[Hashable, float]]:
    list_states = R.get_states()
    list_actions = list(T.keys())
    q_values = {s: {a: 0 for a in list_actions} for s in list_states}
    v = {s: 0 for s in list_states}

    def _successor_value(s, a):
        _sum = 0
        for sp, p in T[a](s).items():
            _sum += p * v[sp]
        return _sum

    for _ in range(iterations):
        for s in list_states:
            for a in list_actions:
                q_values[s][a] = R(s, a) + gamma * _successor_value(s, a)

        # update value function
        for s, qs in q_values.items():
            v[s] = max(qs.values())

    return q_values, v


class ModelBasedAgent:
    terminal_state: str = "terminal"

    def __init__(
        self, n_actions: int = 4, gamma: float = 0.95, default_reward: float = 0.0
    ):
        self.transition_model = TabularStateActionTransitionEstimator(
            n_actions, terminal_state=self.terminal_state
        )
        self.reward_model = TabularRewardEstimator(
            n_actions, terminal_state=self.terminal_state, default_reward=default_reward
        )
        self.gamma = gamma

    def update(self, s: Hashable, a: ActType, r: float, sp: Hashable, done: bool):
        self.transition_model.update(s, a, sp, done)
        self.reward_model.update(s, a, r)

        if done:
            for a in range(self.transition_model.n_actions):
                self.reward_model.update(sp, a, 0)

    def estimate_value_function(self, gamma: int | None = None, iterations: int = 500):
        gamma = gamma if gamma is not None else self.gamma
        T = self.transition_model.get_transition_functions()
        R = self.reward_model
        return value_iteration(T, R, gamma, iterations)

    def estimate_graph_laplacian(self, normalized: bool = True):
        return self.transition_model.get_graph_laplacian(normalized)

    def sample(self, s: Hashable, a: ActType) -> Hashable:
        sp = self.transition_model.sample(s, a)
        r = self.reward_model.sample(s, a)
        return r, sp

    def reset(self):
        self.transition_model.reset()
        self.reward_model.reset()


class ModelFreeAgent:
    def __init__(
        self,
        n_actions: int = 4,
        learning_rate: float = 0.1,
        default_value: float = 0.0,
        gamma: float = 0.99,
    ):
        self.q_values = {}
        self.eta = learning_rate
        self.n_actions = n_actions
        self.gamma = gamma
        self.q_init = {a: default_value for a in range(n_actions)}

    def maybe_init_q_values(self, s: int) -> None:
        """
        Initialize a set of q_values if they don't exist, setting them to
        a default value
        """

        if s not in self.q_values:
            if self.q_values:
                q_init = {
                    a: max([max(v.values()) for v in self.q_values.values()])
                    for a in range(self.n_actions)
                }
            else:
                q_init = self.q_init

            self.q_values[s] = q_init

    def get_q_values(self, s: int) -> np.ndarray:
        def _get_q(s0):
            self.maybe_init_q_values(s0)
            q = np.array(list(self.q_values.get(s0, None).values()))
            return q

        q_values = np.stack([_get_q(s0) for s0 in s])
        return q_values

    def get_value_function(self) -> Dict[int, float]:
        return {s: max(v.values()) for s, v in self.q_values.items()}

    def update(self, s: Hashable, a: ActType, r: float, sp: Hashable):
        self.maybe_init_q_values(s)
        self.maybe_init_q_values(sp)

        q = self.q_values[s][a]
        V = max(self.q_values[sp].values())

        self.q_values[s][a] = q + self.eta * (r + self.gamma * V - q)

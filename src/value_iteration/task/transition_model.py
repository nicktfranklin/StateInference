from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .utils import ActType, StateType, get_position_from_state


def one_hot(a, num_classes):
    ### define simple deterministic transition functions using cardinal movements
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


class TransitionModel:
    # Generative model. Assumes connectivity between neighboring states in a 2d gridworld

    action_keys = ["up", "down", "left", "right"]

    def __init__(
        self,
        h: int,
        w: int,
        walls: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        self.random_transitions = self._make_random_transitions(h, w)
        self.state_action_transitions = self._make_cardinal_transition_function(
            h, w, walls
        )
        self.adjecency_list = self._make_adjecency_list(self.random_transitions)
        self.walls = walls
        self.n_states = h * w
        self.h = h
        self.w = w

    def get_state_action_transitions(
        self, terminal_states: List[int] | None = None
    ) -> np.ndarray:

        if terminal_states is None:
            return self.state_action_transitions

        # Add terminal transitions
        t = np.zeros((len(self.action_keys), self.n_states + 1, self.n_states + 1))
        t[:, :-1, :-1] = self.state_action_transitions
        t[:, -1, -1] = 1  # terminal state
        for s in terminal_states:
            t[:, s, :-1] = 0
            t[:, s, -1] = 1

        return t

    @staticmethod
    def _make_cardinal_transition_function(
        h: int,
        w: int,
        walls: Optional[list[tuple[int, int]]] = None,
    ) -> np.ndarray:
        states = np.arange(h * w).reshape(h, w)
        transitions = [
            np.vstack([states[0, :], states[:-1, :]]),
            np.vstack([states[1:, :], states[-1, :]]),
            np.hstack([states[:, :1], states[:, :-1]]),
            np.hstack([states[:, 1:], states[:, -1:]]),
        ]
        transitions = [one_hot(t.reshape(-1), h * w) for t in transitions]

        if walls is not None:

            def filter_transition_function(t):
                def _filter_wall(b1, b2):
                    t[b1][b1], t[b1][b2] = t[b1][b1] + t[b1][b2], 0
                    t[b2][b2], t[b2][b1] = t[b2][b2] + t[b2][b1], 0

                for b1, b2 in walls:
                    _filter_wall(b1, b2)
                return t

            transitions = [filter_transition_function(t) for t in transitions]

        return np.array(transitions)

    @staticmethod
    def _make_random_transitions(
        h: int, w: int, walls: Optional[list[tuple[StateType, StateType]]] = None
    ) -> np.ndarray:
        t = np.zeros((h * w, h * w))
        for s0 in range(h * w):
            # down
            if s0 - w >= 0:
                t[s0, s0 - w] = 1
            # up
            if s0 + w + 1 < h * w:
                t[s0, s0 + w] = 1

            # right
            if (s0 + 1) % w > 0:
                t[s0, s0 + 1] = 1

            # left
            if s0 % w > 0:
                t[s0, s0 - 1] = 1

        # normalize
        t = TransitionModel._normalize(t)
        if walls:
            return TransitionModel._add_walls(t, walls)
        return t

    @staticmethod
    def _add_walls(
        transitions: dict[StateType, StateType],
        walls: list[tuple[StateType, StateType]],
    ) -> dict[StateType, StateType]:
        for s, sp in walls:
            transitions[s, sp] = 0
            transitions[sp, s] = 0
        return TransitionModel._normalize(transitions)

    @staticmethod
    def _normalize(t: np.ndarray) -> np.ndarray:
        return t / np.tile(t.sum(axis=1).reshape(-1, 1), t.shape[1])

    @staticmethod
    def _make_adjecency_list(transitions: np.ndarray) -> Dict[int, np.ndarray]:
        edges = {}
        for s, t in enumerate(transitions):
            edges[s] = np.where(t > 0)[0]
        return edges

    def get_sucessor_distribution(
        self, state: StateType, action: ActType
    ) -> np.ndarray:
        assert state in self.adjecency_list
        assert action < self.state_action_transitions.shape[0], f"action {action}"
        return self.state_action_transitions[action, state, :]

    def display_gridworld(
        self, ax: Optional[matplotlib.axes.Axes] = None, wall_color="k"
    ) -> matplotlib.axes.Axes:
        if not ax:
            _, ax = plt.subplots(figsize=(5, 5))
            ax.invert_yaxis()

        ax.set_yticks([])
        ax.set_xticks([])
        # plot the gridworld tiles
        for r in range(self.h):
            ax.plot([-0.5, self.w - 0.5], [r - 0.5, r - 0.5], c="grey", lw=0.5)
        for c in range(self.w):
            ax.plot([c - 0.5, c - 0.5], [-0.5, self.h - 0.5], c="grey", lw=0.5)

        if self.walls is None:
            return ax
        for s0, s1 in self.walls:
            r0, c0 = get_position_from_state(s0, self.w)
            r1, c1 = get_position_from_state(s1, self.w)

            x = (r0 + r1) / 2
            y = (c0 + c1) / 2

            assert (r0 == r1) or (c0 == c1), f"Not a valid wall! {r0} {r1} {c0} {s1}"
            if c0 == c1:
                ax.plot([y - 0.5, y + 0.5], [x, x], c=wall_color, lw=3)
            else:
                ax.plot([y, y], [x - 0.5, x + 0.5], c=wall_color, lw=3)

        return ax

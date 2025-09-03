from typing import Any, Dict, List, Optional, SupportsFloat

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..value_iteration.models.value_iteration_network import ValueIterationNetwork
from .observation_model import ObservationModel
from .reward_model import RewardModel
from .thread_the_needle import make_thread_the_needle_walls
from .transition_model import TransitionModel
from .utils import ActType, ObsType, RewType, StateType

OutcomeTuple = tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]


class GridWorldEnv(gym.Env):
    def __init__(
        self,
        transition_model: TransitionModel,
        reward_model: RewardModel,
        observation_model: ObservationModel,
        initial_state: Optional[int] = None,
        n_states: Optional[int] = None,
        end_state: Optional[list[int]] = None,
        random_initial_state: bool = True,
        max_steps: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.observation_model = observation_model

        # self.n_states = n_states if n_states else self.transition_model.n_states
        self.n_states = n_states
        self.initial_state = initial_state
        assert (
            random_initial_state is not None or initial_state is not None
        ), "must specify either the inital location or random initialization"
        self.current_state = self._get_initial_state()

        self.observation = self.generate_observation(self.current_state)
        self.states = np.arange(n_states)
        self.end_state = end_state
        self.step_counter = 0
        self.max_steps = max_steps

        # attributes for gym.Env
        # See: https://gymnasium.farama.org/api/env/
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.observation_model.map_height,
                self.observation_model.map_height,
            ),
            dtype=np.int32,
        )
        self.metadata = None
        self.render_mode = None
        self.reward_range = self.reward_model.get_rew_range()
        self.spec = None

    def _check_truncate(self) -> bool:
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            return True
        return False

    def _check_terminate(self, state: int) -> bool:
        if self.end_state == None:
            return False
        return state in self.end_state

    def _get_initial_state(self) -> int:
        if self.initial_state:
            return self.initial_state
        return np.random.randint(self.n_states)

    def display_gridworld(
        self,
        ax: Optional[matplotlib.axes.Axes] = None,
        wall_color="k",
        annotate: bool = True,
    ) -> matplotlib.axes.Axes:
        if not ax:
            _, ax = plt.subplots(figsize=(5, 5))
            ax.invert_yaxis()
        self.transition_model.display_gridworld(ax, wall_color)

        if annotate:
            for s, rew in self.reward_model.successor_state_rew.items():
                loc = self.observation_model.get_grid_coords(s)
                c = "b" if rew > 0 else "r"
                ax.annotate(f"{rew}", loc, ha="center", va="center", c=c)
        ax.set_title("Thread-the-needle states")
        return ax

    def generate_observation(self, state: int) -> np.ndarray:
        return self.observation_model(state)

    def get_obseservation(self) -> np.ndarray:
        return self.observation

    def get_state(self) -> np.ndarray:
        return self.current_state

    def set_initial_state(self, initial_state):
        self.initial_state = initial_state

    def get_state_values(
        self, outcomes: List[OutcomeTuple] | None = None, gamma: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            sars: List of (state, action, reward, next_state) tuples
        Returns:
            np.ndarray: state values
        """
        if outcomes is None:

            t = self.transition_model.get_state_action_transitions(self.end_state)
            r = self.reward_model.construct_rew_func(t)

            # yet another value iteration implementation
            n_actions, n_states, _ = t.shape
            q = np.zeros((n_states, n_actions))
            v = np.zeros(n_states)
            for _ in range(1_000):
                q = r + gamma * np.dot(t, v)
                v = q.max(axis=0)

            return q, v
        else:
            ### assume we know the cardinality of the state space
            raise NotImplementedError

    def get_optimal_policy(self, gamma=0.99) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            pi: np.ndarray
                policy, shape (n_states, n_actions)
            values: np.ndarray
                state values, shape (n_states,)
        """

        t = self.transition_model.get_state_action_transitions(self.end_state)
        r = self.reward_model.construct_rew_func(t)

        # yet another value iteration implementation
        n_actions, n_states, _ = t.shape
        q = np.zeros((n_states, n_actions))
        v = np.zeros(n_states)
        for _ in range(1_000):
            q = r + gamma * np.dot(t, v)
            v = q.max(axis=0)

        # remove the terminal state
        q = q[:, :-1].T

        return (
            np.array([np.isclose(q0, q0.max()) for q0 in q], dtype=float),
            v[:-1],
        )

    # Key methods from Gymnasium:
    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        if options is not None:
            assert isinstance(options, dict)
            initial_state = options.get("initial_state", None)
        else:
            initial_state = None

        self.current_state = (
            initial_state if initial_state is not None else self._get_initial_state()
        )
        self.step_counter = 0

        return self.generate_observation(self.current_state), dict()

    def step(self, action: ActType) -> OutcomeTuple:
        self.step_counter += 1

        pdf_s = self.transition_model.get_sucessor_distribution(
            self.current_state, action
        )

        assert np.sum(pdf_s) == 1, (action, self.current_state, pdf_s)
        assert np.all(pdf_s >= 0), print(pdf_s)

        successor_state = np.random.choice(self.states, p=pdf_s)
        sucessor_observation = self.generate_observation(successor_state)

        reward = self.reward_model.get_reward(successor_state)
        terminated = self._check_terminate(successor_state)
        truncated = self._check_truncate()
        info = dict(start_state=self.current_state, successor_state=successor_state)

        self.current_state = successor_state
        self.observation = sucessor_observation

        output = tuple([sucessor_observation, reward, terminated, truncated, info])

        return output

    def render(self) -> None:
        raise NotImplementedError

    def close(self):
        pass


class ThreadTheNeedleEnv(GridWorldEnv):
    @classmethod
    def create_env(
        cls,
        height: int,
        width: int,
        map_height: int,
        state_rewards: dict[StateType, RewType],
        observation_kwargs: dict[str, Any],
        movement_penalty: float = 0.0,
        **gridworld_env_kwargs,
    ):
        # Define the transitions for the thread the needle task
        walls = make_thread_the_needle_walls(20)
        transition_model = TransitionModel(height, width, walls)

        observation_model = ObservationModel(
            height, width, map_height, **observation_kwargs
        )

        reward_model = RewardModel(state_rewards, movement_penalty)

        return cls(
            transition_model, reward_model, observation_model, **gridworld_env_kwargs
        )


class OpenEnv(ThreadTheNeedleEnv):
    @classmethod
    def create_env(
        cls,
        height: int,
        width: int,
        map_height: int,
        state_rewards: dict[StateType, RewType],
        observation_kwargs: dict[str, Any],
        movement_penalty: float = 0.0,
        **gridworld_env_kwargs,
    ):
        # Define the transitions for the thread the needle task
        transition_model = TransitionModel(height, width, None)

        observation_model = ObservationModel(
            height, width, map_height, **observation_kwargs
        )

        reward_model = RewardModel(state_rewards, movement_penalty)

        return cls(
            transition_model, reward_model, observation_model, **gridworld_env_kwargs
        )


class CnnWrapper(ThreadTheNeedleEnv):
    def __init__(self, env):
        self.parent = env
        self.parent.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.observation_model.map_height,
                self.observation_model.map_height,
                1,
            ),
            dtype=np.uint8,
        )

    def __getattr__(self, attr):
        return getattr(self.parent, attr)

    def generate_observation(self, state: int) -> np.ndarray:
        return self.parent.generate_observation(state).reshape(
            self.observation_model.map_height, self.observation_model.map_height, 1
        )

from typing import Any, Dict, Optional, TypeVar

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from task.observation_model import ObservationModel
from task.reward_model import RewardModel
from task.transition_model import TransitionModel
from task.utils import ActType, ObsType, RewType, StateType
from value_iteration.environments.thread_the_needle import make_thread_the_needle_walls
from value_iteration.models.value_iteration_network import ValueIterationNetwork


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

    def _check_terminate(self, state: int) -> bool:
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            return True
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

    def get_optimal_policy(self) -> np.ndarray:
        t = self.transition_model.state_action_transitions
        r = self.reward_model.construct_rew_func(t)
        sa_values, values = ValueIterationNetwork.value_iteration(
            t,
            r,
            self.observation_model.h,
            self.observation_model.w,
            gamma=0.8,
            iterations=1000,
        )
        return (
            np.array([np.isclose(v, v.max()) for v in sa_values], dtype=float),
            values,
        )

    # Key methods from Gymnasium:
    def reset(self, seed=None, options=None) -> tuple[np.ndarray, Dict[str, Any]]:
        self.current_state = self._get_initial_state()
        self.step_counter = 0

        return self.generate_observation(self.current_state), dict()

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, Dict[str, Any]]:
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
        truncated = False
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

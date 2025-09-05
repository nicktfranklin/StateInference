from typing import List, Tuple, Union

import numpy as np
from scipy import sparse
from tqdm.notebook import trange

from value_iteration.models.utils import softmax


class PlanningModel:
    def __init__(
        self,
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        reward_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        n_rows: int,
        n_columns: int,
        n_actions: int = 4,
        gamma: float = 0.8,
        beta: float = 1.0,
    ):
        self.transition_functions = transition_functions
        self.reward_functions = reward_functions
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_actions = n_actions
        self.gamma = gamma
        self.beta = beta

        assert (self.gamma > 0) and (
            self.gamma < 1
        ), "gamma must be greater than zero and less than 1"

        assert beta > 0, "Beta must be strictly positive!"

    def inference(self):
        raise NotImplementedError


class ValueIterationNetwork(PlanningModel):
    def __init__(
        self,
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        reward_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        n_rows: int,
        n_columns: int,
        n_actions: int = 4,
        gamma: float = 0.8,
        beta: float = 1.0,
        initialization_noise: float = 0.01,
    ):
        PlanningModel.__init__(
            self,
            transition_functions,
            reward_functions,
            n_rows,
            n_columns,
            n_actions,
            gamma,
            beta,
        )
        self.initialization_noise = initialization_noise

    def inference(self, iterations: int = 100):
        state_action_values, _ = self.value_iteration(
            self.transition_functions,
            self.reward_functions,
            self.n_rows,
            self.n_columns,
            self.gamma,
            iterations,
            self.initialization_noise,
        )
        return softmax(state_action_values, self.beta)

    @staticmethod
    def value_iteration(
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        reward_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        n_rows: int,
        n_columns: int,
        gamma: float = 0.8,
        iterations: int = 10,
        initialization_noise: float = 0.01,  # in standard deviation
        return_interim_estimates: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For the reward, transition matrix, list indicies correspond to actions.  E.g.
        the 0th action will have the 0the transition/reward function.

        This assumes a square-grid structure
        """

        tiling = ValueIterationNetwork.define_lattice_tiling(n_rows, n_columns)
        n_states = n_rows * n_columns
        n_actions = len(reward_functions)

        # initialize the value functions
        value_function = ValueIterationNetwork._initialize_value_function(
            n_rows, n_columns, initialization_noise
        )
        value_0 = value_function[tiling]
        value_1 = value_function[~tiling]

        state_action_values = np.zeros((n_states, n_actions))  # q(s, a)
        q_0, q_1 = (
            [np.empty((n_states // 2, n_actions))] * n_actions,
            [np.empty((n_states // 2, n_actions))] * n_actions,
        )

        if return_interim_estimates:
            state_action_values = np.zeros((iterations, n_states, n_actions))
            value_function = np.zeros((iterations, n_states))

        # dynamic programing algorithm
        for ii in trange(0, iterations):
            # calculate tiled values (we could randomize the tiling order but it isn't important)
            for a, (r_a, t_a) in enumerate(zip(reward_functions, transition_functions)):
                q_0[a] = r_a[tiling] + gamma * t_a[tiling][:, ~tiling].dot(value_1)
            value_0 = np.max(q_0, axis=0)

            for a, (r_a, t_a) in enumerate(zip(reward_functions, transition_functions)):
                q_1[a] = r_a[~tiling] + gamma * t_a[~tiling][:, tiling].dot(value_0)
            value_1 = np.max(q_1, axis=0)

            if return_interim_estimates:
                value_function[ii, tiling] = value_0
                value_function[ii, ~tiling] = value_1
                state_action_values[ii, tiling, :] = np.array(q_0).T
                state_action_values[ii, ~tiling, :] = np.array(q_1).T

        if return_interim_estimates:
            return state_action_values, value_function

        value_function[tiling] = value_0
        value_function[~tiling] = value_1

        state_action_values[tiling, :] = np.array(q_0).T
        state_action_values[~tiling, :] = np.array(q_1).T

        return state_action_values, value_function

    @staticmethod
    def define_lattice_tiling(n_rows: int, n_columns: int) -> np.ndarray:
        row_order_a = [ii % 2 for ii in range(n_columns)]
        row_order_b = [ii % 2 for ii in range(1, n_columns + 1)]
        tiling = []
        for ii in range(n_rows):
            if ii % 2 == 0:
                tiling += row_order_a
            else:
                tiling += row_order_b

        return np.array(tiling, dtype=bool)

    @staticmethod
    def _initialize_value_function(
        n_rows: int, n_columns: int, noise: float
    ) -> np.ndarray:
        return np.random.normal(loc=0, scale=noise, size=(n_rows, n_columns)).reshape(
            -1
        )


class UntiledValueIterationNetwork(ValueIterationNetwork):
    @staticmethod
    def value_iteration(
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        reward_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        n_rows: int,
        n_columns: int,
        gamma: float = 0.8,
        iterations: int = 10,
        initialization_noise: float = 0.01,  # in standard deviation
        return_interim_estimates: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For the reward, transition matrix, list indicies correspond to actions.  E.g.
        the 0th action will have the 0the transition/reward function.

        This assumes a square-grid structure
        """

        n_states = n_rows * n_columns
        n_actions = len(reward_functions)

        # initialize the value functions
        value_function = ValueIterationNetwork._initialize_value_function(
            n_rows, n_columns, initialization_noise
        )
        value = value_function

        state_action_values = np.zeros((n_states, n_actions))  # q(s, a)
        q = [np.empty((n_states, n_actions))] * n_actions

        if return_interim_estimates:
            state_action_values = np.zeros((iterations, n_states, n_actions))
            value_function = np.zeros((iterations, n_states))

        # dynamic programing algorithm
        for ii in trange(0, iterations):
            # calculate values
            for a, (r_a, t_a) in enumerate(zip(reward_functions, transition_functions)):
                q[a] = r_a + gamma * t_a.dot(value)
            value = np.max(q, axis=0)

            if return_interim_estimates:
                value_function[ii, :] = value
                state_action_values[ii, ...] = np.array(q).T

        if return_interim_estimates:
            return state_action_values, value_function

        value_function = value

        state_action_values = np.array(q).T

        return state_action_values, value_function

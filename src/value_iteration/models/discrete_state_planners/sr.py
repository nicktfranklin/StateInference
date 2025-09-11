from random import sample
from typing import List, Optional, Union

import numpy as np
import scipy
from scipy.special import logsumexp
from tqdm import tqdm

from value_iteration.models.discrete_state_planners.value_iteration_network import (
    ValueIterationNetwork,
)
from value_iteration.models.utils import (
    calculate_sr_from_transitions,
    inverse_cmf_sampler,
    one_hot,
)


def get_state_action_reward_from_sucessor_rewards(
    reward_function_over_sucessors: np.ndarray,
    transitions: List[Union[np.ndarray, scipy.sparse.csr_matrix]],
) -> List[Union[np.ndarray, scipy.sparse.csr_matrix]]:
    reward_function_over_sa = [
        t.dot(reward_function_over_sucessors) for t in transitions
    ]
    return reward_function_over_sa


def get_optimal_sr_from_transitions(
    transition_functions,
    optimal_policy,
    gamma=0.95,
):
    # turn optimal policy in to pmf
    pi = optimal_policy * np.tile(1 / optimal_policy.sum(axis=1), (4, 1)).T

    # marginalize out the optimal policy
    T_sas = np.transpose(transition_functions, (1, 0, 2))
    T_ss = np.array([T_sas[ii, :].T.dot(pi[ii]) for ii in range(pi.shape[0])])

    # calculate the SR
    return calculate_sr_from_transitions(T_ss, gamma)


def get_value_function_from_sr(
    successor_representation: np.ndarray, state_reward_function: np.ndarray
) -> np.ndarray:
    return successor_representation.dot(state_reward_function)


class SRResampler:
    """
    This version of the SR sampling mechanism comes from Stachenfeld, 2017.  Here,
    we assume randomly sampled transitions from the True generative process.

    This needs to be framed as a simplified variant of the Russek model,
    or maybe even a new model we propose.
    """

    def __init__(
        self,
        learning_rate: float,
        gamma: float,
        transition_functions: List[np.ndarray],
        state_reward_function: np.ndarray,
        beta: float = 1,
        n_rows: Optional[int] = None,
        n_columns: Optional[int] = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.transition_functions = transition_functions
        self.reward_function = state_reward_function
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_states = self.transition_functions[0].shape[0]
        self.n_actions = len(self.transition_functions)
        self.beta = beta

        self.optimal_value_function: Optional[np.ndarray] = None

    def _calculate_optimal_value_function(
        self,
        iterations: int = 1000,
        n_rows: Optional[int] = None,
        n_columns: Optional[int] = None,
    ) -> np.ndarray:
        assert n_rows or self.n_rows, "n_rows not defined"
        assert n_columns or self.n_columns, "n_colmuns not defined"

        state_action_reward_functions = get_state_action_reward_from_sucessor_rewards(
            self.reward_function, self.transition_functions
        )

        _, state_value_function = ValueIterationNetwork.value_iteration(
            transition_functions=self.transition_functions,
            reward_functions=state_action_reward_functions,
            n_rows=self.n_rows,
            n_columns=self.n_columns,
            gamma=self.gamma,
            iterations=iterations,
            return_interim_estimates=False,
        )

        return state_value_function

    def _calculate_optimal_sr(self) -> None:
        pass

    def _delta_update(
        self,
        initial_state: int,
        successor_state: int,
        sr: np.ndarray,
    ) -> np.ndarray:
        sr = sr.copy()
        sr[initial_state, :] = sr[initial_state, :] + self.learning_rate * (
            one_hot(initial_state, self.n_states)
            + self.gamma * sr[successor_state, :]
            - sr[initial_state, :]
        )

        return sr

    def _sample_successor_state(self, inital_state: int, action: int) -> int:
        return inverse_cmf_sampler(self.transition_functions[action][inital_state, :])

    def _sample_action_from_value_function(
        self,
        state: int,
        value_function: np.ndarray,
    ) -> int:
        # marginalize out the successor state for each action
        q_values = np.array(
            [
                self.gamma * self.transition_functions[a][state, :].dot(value_function)
                for a in range(self.n_actions)
            ]
        )

        log_pmf = self.beta * q_values - logsumexp(self.beta * q_values)
        return inverse_cmf_sampler(np.exp(log_pmf))

    def _one_sample_update(
        self, successor_representation: np.ndarray, value_function: np.ndarray
    ) -> np.ndarray:
        sampled_initial_state = np.random.randint(self.n_states)

        # sample according to precomputed value function
        sampled_action = self._sample_action_from_value_function(
            sampled_initial_state,
            value_function,
        )
        sampled_successor_state = self._sample_successor_state(
            sampled_initial_state, sampled_action
        )

        return self._delta_update(
            sampled_initial_state, sampled_successor_state, successor_representation
        )

    @staticmethod
    def get_value_function(
        successor_representation: np.ndarray, state_reward_function: np.ndarray
    ) -> np.ndarray:
        return get_value_function_from_sr(
            successor_representation, state_reward_function
        )

    @staticmethod
    def convert_state_value_to_q(
        transition_functions: np.ndarray, state_value_function: np.ndarray, gamma: float
    ) -> List[np.ndarray]:
        # Q(s, a) = T(s, a, s')V(s')

        n_actions = len(transition_functions)

        q_values = [
            gamma * transition_functions[a].dot(state_value_function)
            for a in range(n_actions)
        ]

        return q_values

    @staticmethod
    def get_q_values(
        transition_functions: np.ndarray,
        successor_representation: np.ndarray,
        state_reward_function: np.ndarray,
        gamma: float = 1,
    ) -> List[np.ndarray]:
        # Q(s, a) = T(s, a, s')V(s')

        state_value_function = get_value_function_from_sr(
            successor_representation, state_reward_function
        )

        return SRResampler.convert_state_value_to_q(
            transition_functions, state_value_function, gamma
        )

    def _resample_step(
        self, sr: np.ndarray, state_value_function: np.ndarray
    ) -> np.ndarray:
        # sample each state w/o replacement
        for sampled_state in sample(range(self.n_states), self.n_states):
            sr = self._one_sample_update(sr, state_value_function)

            # sample according to precomputed value function
            sampled_action = self._sample_action_from_value_function(
                sampled_state,
                state_value_function,
            )
            sampled_successor_state = self._sample_successor_state(
                sampled_state, sampled_action
            )

            sr = self._delta_update(sampled_state, sampled_successor_state, sr)

        return sr

    def resample(
        self,
        initial_sr: np.ndarray,
        state_values: Optional[np.ndarray] = None,
        restimate_state_values: bool = True,
        return_iterim_estimates: bool = False,
        steps: int = 100,
    ) -> Union[List[np.ndarray], np.ndarray]:
        assert (
            state_values is not None or restimate_state_values
        ), "Must pass a state value function if not reestimating"

        estimates = []

        sr_est = initial_sr.copy()

        for _ in tqdm(range(steps), desc="Resampling"):
            if restimate_state_values:
                state_values = get_value_function_from_sr(sr_est, self.reward_function)

            sr_est = self._resample_step(sr_est, state_values)

            if return_iterim_estimates:
                estimates.append(sr_est)

        if return_iterim_estimates:
            return estimates

        return sr_est

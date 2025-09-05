from typing import Union

import numpy as np
import scipy.sparse
from scipy.special import logsumexp


def inverse_cmf_sampler(pmf: Union[np.ndarray, scipy.sparse.csr_matrix]) -> int:
    if type(pmf) == scipy.sparse.csr_matrix:
        pmf = pmf.toarray()

    return np.array(np.cumsum(np.array(pmf)) < np.random.rand(), dtype=int).sum()


def sample_trajectory_until_goal(
    start_state: int, goal_state: int, policy: np.ndarray, transition_functions
):
    current_state = start_state
    state_trajectory = [current_state]
    while current_state != goal_state:
        sampled_action = inverse_cmf_sampler(policy[current_state, :])

        t = transition_functions[sampled_action]

        current_state = inverse_cmf_sampler(t[current_state, :])
        state_trajectory.append(current_state)

    return state_trajectory


def evaluate_policy(
    policy: np.ndarray,
    optimal_policy: np.ndarray,
) -> np.ndarray:
    """

    :param policy: binary vector of state-action, 1 if in optimal policy, zero otherwise (not pmf)
    :param optimal_policy: pmf
    :return: vector of errors, 1-per state
    """

    return 1 - np.sum(optimal_policy * policy, axis=1)


def softmax(state_action_values: np.ndarray, beta: float = 1) -> np.ndarray:
    assert beta > 0, "Beta must be strictly positive!"

    def _internal_softmax(q: np.ndarray) -> np.ndarray:
        return np.exp(beta * q - logsumexp(beta * q))

    if np.ndim(state_action_values) == 1:
        return _internal_softmax(state_action_values)

    return np.array(list(map(_internal_softmax, state_action_values)))


def one_hot(x, depth: int):
    return np.take(np.eye(depth), x, axis=0)


def calculate_sr_from_transitions(
    transition_function: Union[np.ndarray, scipy.sparse.csr_matrix], gamma: float
) -> np.ndarray:
    n = transition_function.shape[0]
    return np.linalg.inv(np.eye(n) - gamma * transition_function)

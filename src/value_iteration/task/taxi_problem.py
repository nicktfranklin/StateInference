from typing import List, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix


def pos_to_state(row: int, col: int, counter: int, grid_size: int) -> int:
    """Convert (row, col, counter) to state index.

    State encoding:
    - Layer 0: counter=0 - states 0 to grid_size²-1
    - Layer 1: counter=1 - states grid_size² to 2*grid_size²-1
    - Layer 2: counter=2 - states 2*grid_size² to 3*grid_size²-1
    """
    layer_offset = counter * grid_size * grid_size
    return layer_offset + row * grid_size + col


def state_to_pos(state: int, grid_size: int) -> Tuple[int, int, int]:
    """Convert state index to (row, col, counter)."""
    states_per_layer = grid_size * grid_size
    counter = state // states_per_layer
    remaining = state % states_per_layer
    row = remaining // grid_size
    col = remaining % grid_size

    return row, col, counter


def get_valid_neighbors(row: int, col: int, grid_size: int) -> List[Tuple[int, int]]:
    """Get valid 4-directional neighbors within grid bounds."""
    neighbors = []
    # North, South, East, West
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
            neighbors.append((new_row, new_col))

    return neighbors


class TaxiRewardFunction:
    """Modular reward function system for taxi problem."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.total_states = 3 * grid_size * grid_size
        self.rewards = {}

    def set_reward(self, row: int, col: int, counter: int, reward: float) -> None:
        """Set reward for a specific state configuration."""
        state = pos_to_state(row, col, counter, self.grid_size)
        self.rewards[state] = reward

    def set_position_reward(
        self, row: int, col: int, reward: float, all_counter_states: bool = True
    ) -> None:
        """Set reward for a position across all or specific counter states."""
        if all_counter_states:
            for counter in [0, 1, 2]:
                self.set_reward(row, col, counter, reward)

    def get_reward_vector(self, default_reward: float = -0.01) -> np.ndarray:
        """Generate reward vector with default values for unspecified states."""
        reward_vector = np.full(self.total_states, default_reward)

        for state, reward in self.rewards.items():
            reward_vector[state] = reward

        return reward_vector


def create_taxi_transition_functions(
    grid_size: int,
    stop1_pos: Tuple[int, int],
    stop2_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    slip_probability: float = 0.05,
    sparse: bool = True,
) -> List[Union[np.ndarray, csr_matrix]]:
    """Create transition functions for taxi problem with counter-based stops.

    Actions: [North=0, South=1, East=2, West=3]
    Counter increments: 0->1 at stop1_pos, 1->2 at stop2_pos
    """
    total_states = 3 * grid_size * grid_size
    n_actions = 4

    # Initialize transition matrices for each action
    if sparse:
        transitions = [
            csr_matrix((total_states, total_states)) for _ in range(n_actions)
        ]
    else:
        transitions = [np.zeros((total_states, total_states)) for _ in range(n_actions)]

    # Direction mappings: North, South, East, West
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for state in range(total_states):
        row, col, counter = state_to_pos(state, grid_size)

        # Special handling for goal state when counter is 2 (both stops completed)
        if (row, col) == goal_pos and counter == 2:
            # Goal state is absorbing
            for action in range(n_actions):
                if sparse:
                    transitions[action] = transitions[action] + csr_matrix(
                        ([1.0], ([state], [state])), shape=(total_states, total_states)
                    )
                else:
                    transitions[action][state, state] = 1.0
            continue

        for action in range(n_actions):
            # Calculate intended next position
            dr, dc = directions[action]
            next_row, next_col = row + dr, col + dc

            # Check if movement is valid (within bounds)
            if 0 <= next_row < grid_size and 0 <= next_col < grid_size:
                intended_row, intended_col = next_row, next_col
            else:
                # Invalid move, stay in place
                intended_row, intended_col = row, col

            # Handle counter increment logic
            new_counter = counter

            # Counter increment logic - increment based on stop visits
            if (intended_row, intended_col) == stop1_pos and counter == 0:
                new_counter = 1
            elif (intended_row, intended_col) == stop2_pos and counter == 1:
                new_counter = 2

            # Calculate next state with potential counter increment
            next_state = pos_to_state(
                intended_row,
                intended_col,
                new_counter,
                grid_size,
            )

            # Apply slip probability
            successful_prob = 1.0 - slip_probability
            slip_prob_per_direction = slip_probability / 4.0

            # Main intended transition
            if sparse:
                transitions[action] = transitions[action] + csr_matrix(
                    ([successful_prob], ([state], [next_state])),
                    shape=(total_states, total_states),
                )
            else:
                transitions[action][state, next_state] += successful_prob

            # Slip transitions (try all 4 directions with equal probability)
            for slip_action in range(n_actions):
                slip_dr, slip_dc = directions[slip_action]
                slip_next_row, slip_next_col = row + slip_dr, col + slip_dc

                # Check bounds for slip movement
                if 0 <= slip_next_row < grid_size and 0 <= slip_next_col < grid_size:
                    slip_intended_row, slip_intended_col = slip_next_row, slip_next_col
                else:
                    slip_intended_row, slip_intended_col = row, col

                # Apply same counter increment logic for slip movements
                slip_counter = counter

                if (slip_intended_row, slip_intended_col) == stop1_pos and counter == 0:
                    slip_counter = 1
                elif (
                    slip_intended_row,
                    slip_intended_col,
                ) == stop2_pos and counter == 1:
                    slip_counter = 2

                slip_next_state = pos_to_state(
                    slip_intended_row,
                    slip_intended_col,
                    slip_counter,
                    grid_size,
                )

                if sparse:
                    transitions[action] = transitions[action] + csr_matrix(
                        ([slip_prob_per_direction], ([state], [slip_next_state])),
                        shape=(total_states, total_states),
                    )
                else:
                    transitions[action][
                        state, slip_next_state
                    ] += slip_prob_per_direction

    return transitions


def create_taxi_optimal_policy_placeholder(
    grid_size: int,
    agent_pos: Tuple[int, int],
    stop1_pos: Tuple[int, int],
    stop2_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
) -> np.ndarray:
    """Create placeholder optimal policy for taxi problem.

    This is a simplified heuristic policy, not the true optimal policy.
    The true optimal policy would need to be computed via value iteration.
    """
    total_states = 3 * grid_size * grid_size
    n_actions = 4

    # Initialize with uniform random policy
    optimal_policy = np.ones((total_states, n_actions), dtype=float)

    # Normalize to create valid probability distributions
    optimal_policy = optimal_policy / n_actions

    return optimal_policy


def make_gridworld_taxi(
    grid_size: int = 10,
    agent_pos: Tuple[int, int] = (0, 0),
    stop1_pos: Tuple[int, int] = (4, 5),
    stop2_pos: Tuple[int, int] = (7, 2),
    goal_pos: Tuple[int, int] = (9, 9),
    slip_probability: float = 0.05,
    movement_penalty: float = -0.01,
    goal_reward: float = 1.0,
    sparse: bool = True,
) -> Tuple[List[Union[np.ndarray, csr_matrix]], np.ndarray, np.ndarray]:
    """Create taxi gridworld problem compatible with thread_the_needle API.

    Args:
        grid_size: Size of the square grid
        agent_pos: Starting position of taxi (row, col)
        stop1_pos: Position of first stop (row, col)
        stop2_pos: Position of second stop (row, col)
        goal_pos: Final destination position (row, col)
        slip_probability: Probability of taking unintended action
        movement_penalty: Reward for each step (typically negative)
        goal_reward: Reward for reaching goal after both stops
        sparse: Whether to use sparse matrices

    Returns:
        Tuple of (transition_functions, state_reward_function, optimal_policy)
    """
    # Create transition functions
    transition_functions = create_taxi_transition_functions(
        grid_size, stop1_pos, stop2_pos, goal_pos, slip_probability, sparse
    )

    # Create reward function using the modular system
    reward_func = TaxiRewardFunction(grid_size)

    # Set movement penalty for all states
    for row in range(grid_size):
        for col in range(grid_size):
            reward_func.set_position_reward(row, col, movement_penalty)

    # Set goal reward (only when counter is 2 - both stops completed)
    reward_func.set_reward(goal_pos[0], goal_pos[1], 2, goal_reward)

    # Generate final reward vector
    state_reward_function = reward_func.get_reward_vector(movement_penalty)

    # Create placeholder optimal policy
    optimal_policy = create_taxi_optimal_policy_placeholder(
        grid_size, agent_pos, stop1_pos, stop2_pos, goal_pos
    )

    return transition_functions, state_reward_function, optimal_policy


# Utility functions for compatibility with existing visualization and analysis tools
class TaxiGridWorld:
    """Extended GridWorld class for taxi problem."""

    def __init__(
        self,
        grid_size: int,
        transition_functions: List[Union[np.ndarray, csr_matrix]],
        state_reward_function: np.ndarray,
        agent_pos: Tuple[int, int],
        stop1_pos: Tuple[int, int],
        stop2_pos: Tuple[int, int],
        goal_pos: Tuple[int, int],
    ):
        self.grid_size = grid_size
        self.transition_functions = transition_functions
        self.state_reward_function = state_reward_function
        self.agent_pos = agent_pos
        self.stop1_pos = stop1_pos
        self.stop2_pos = stop2_pos
        self.goal_pos = goal_pos

    @staticmethod
    def get_state_from_taxi_position(
        row: int, col: int, counter: int, grid_size: int
    ) -> int:
        """Convert taxi position and counter to state index."""
        return pos_to_state(row, col, counter, grid_size)

    @staticmethod
    def get_taxi_position_from_state(
        state: int, grid_size: int
    ) -> Tuple[int, int, int]:
        """Convert state index to taxi position and counter."""
        return state_to_pos(state, grid_size)


def get_state_action_reward_from_successor_rewards(
    reward_function_over_successors: np.ndarray,
    transitions: List[Union[np.ndarray, csr_matrix]],
) -> List[np.ndarray]:
    """Compute state-action rewards from successor state rewards."""
    if isinstance(transitions[0], csr_matrix):
        reward_function_over_sa = [
            t.dot(reward_function_over_successors) for t in transitions
        ]
    else:
        reward_function_over_sa = [
            t.dot(reward_function_over_successors) for t in transitions
        ]
    return reward_function_over_sa

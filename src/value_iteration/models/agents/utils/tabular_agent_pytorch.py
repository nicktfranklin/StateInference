from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor


def maybe_convert_to_list(x: int | ArrayLike) -> int | list[int]:
    if isinstance(x, int):
        return x
    if isinstance(x, list):
        assert all(isinstance(i, int) for i in x), "Expected list of ints"
        return x
    if isinstance(x, tuple):
        assert all(isinstance(i, int) for i in x), "Expected tuple of ints"
        return list(x)
    if isinstance(x, np.ndarray):
        if len(x) == 1:
            return maybe_convert_to_list(x.item())
        return maybe_convert_to_list(x.tolist())
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return maybe_convert_to_list(x.item())
        return maybe_convert_to_list(x.tolist())
    raise ValueError(f"Expected int or list, got {type(x)}")


def value_iteration(
    transition_function: Tensor,
    reward_function: Tensor,
    gamma: float,
    epsilon: float = 1e-6,
    max_iterations: int = 100,
) -> Tuple[Tensor, Tensor]:
    """Perform value iteration on a Markov Decision Process.

    Args:
        transition_function: Tensor of shape (n_states, n_actions, n_states)
        reward_function: Tensor of shape (n_states, n_actions)
        gamma: Discount factor
        epsilon: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        Tuple[Tensor, Tensor]: Tuple of (Q-values, optimal value function) with shapes
            ((n_states, n_actions), (n_states,))
    """
    n_states, n_actions, _ = transition_function.shape
    q_values = torch.zeros(n_states, n_actions, device=transition_function.device)

    assert gamma >= 0 and gamma < 1

    for _ in range(max_iterations):
        old_q_values = q_values.clone()
        value_function = q_values.max(dim=1).values

        assert value_function.shape == (n_states,)
        assert reward_function.shape == (n_states, n_actions)
        assert transition_function.shape == (n_states, n_actions, n_states)

        q_values = reward_function + gamma * torch.matmul(
            transition_function, value_function
        )

        # Check convergence
        if torch.max(torch.abs(q_values - old_q_values)) < epsilon:
            break

    value_function = q_values.max(dim=1).values
    return q_values, value_function


class TransitionModel:
    def __init__(
        self,
        n_actions: int,
        n_states: int | None = None,
        device: Optional[str] = None,
        default_count: float = 1e-6,
    ):
        assert n_actions is not None
        self.device = device or torch.device("cpu")
        self.default_count = default_count

        # if n_states is None, set it to 1 for the terminal state
        n_states = n_states if n_states is not None else 1

        self.transition_counts = torch.zeros(
            (n_states, n_actions, n_states), device=self.device
        )

        # Initialize counts to go to terminal state, which is always the last state
        self.transition_counts[:, :, -1] = self.default_count
        self.transition_counts[-1, :, -1] = 1.0  # Terminal state is absorbing

        self.state_action_visited = torch.zeros(
            (n_states, n_actions), device=self.device, dtype=torch.bool
        )
        self.state_action_visited[-1, :] = True  # Mark terminal state as visited

    def reset(self, n_states: int | None = None) -> None:
        """Reset the transition model."""
        # initialize with terminal state if no states are provided
        n_states = n_states if n_states is not None else 1
        self.transition_counts = torch.zeros(
            (n_states, self.n_actions, n_states), device=self.device
        )
        self.state_action_visited = torch.zeros(
            (n_states, self.n_actions), device=self.device, dtype=torch.bool
        )
        self.state_action_visited[-1, :] = True  # Mark terminal state as visited

    @property
    def n_actions(self) -> int:
        return self.transition_counts.shape[1]

    @property
    def n_states(self) -> int:
        return self.transition_counts.shape[0]

    @property
    def terminal_state(self) -> int:
        return self.transition_counts.shape[0] - 1

    def update(self, state: int, action: int, next_state: int, done: bool) -> None:
        """Update the transition model with a new observation."""
        if not (0 <= state < self.n_states and 0 <= action < self.n_actions):
            raise ValueError(
                f"Invalid state {state} of {self.n_states} states or action {action} of {self.n_actions} actions"
            )

        if not self.state_action_visited[state, action]:
            # Reset counts only for non-terminal transitions
            self.transition_counts[state, action, :-1] = 0
            # Maintain small probability to terminal
            self.transition_counts[state, action, -1] = self.default_count
            self.state_action_visited[state, action] = True

        # Update next state transition
        if next_state != self.terminal_state:
            self.transition_counts[state, action, next_state] += 1

        # If episode is done, increment terminal state transition
        if done:
            self.transition_counts[state, action, self.terminal_state] += 1

    def check_add_new_state(self, state: int) -> None:
        """Add a new state to the transition model."""
        if state < self.terminal_state:
            return

        n = self.n_states
        new_transition_counts = torch.zeros(
            (n + 1, self.n_actions, n + 1), device=self.device
        )

        # Copy existing transitions
        new_transition_counts[:-2, :, :-2] = self.transition_counts[:-1, :, :-1]

        # Copy terminal transitions to new terminal state
        new_transition_counts[:-2, :, -1] = self.transition_counts[:-1, :, -1]

        # Initialize new state transitions
        new_transition_counts[-2, :, -1] = self.default_count
        # New terminal state is absorbing
        new_transition_counts[-1, :, -1] = 1.0

        # Update state action visited
        new_state_action_visited = torch.zeros(
            (n + 1, self.n_actions), device=self.device, dtype=torch.bool
        )
        new_state_action_visited[:-2, :] = self.state_action_visited[:-1, :]
        new_state_action_visited[-1, :] = True  # Terminal state is always visited

        self.transition_counts = new_transition_counts
        self.state_action_visited = new_state_action_visited

    def get_transition_function(self) -> Tensor:
        """Get the current transition function probabilities."""
        return self.transition_counts / (
            self.transition_counts.sum(dim=-1, keepdim=True) + 1e-6
        )

    def estimate_graph_laplacian(
        self, normalized: bool = True, return_terminal_state: bool = False
    ) -> Tensor:
        """Estimate the graph Laplacian matrix of the transition model."""
        transition_function = self.get_transition_function()

        if not return_terminal_state:
            mask = torch.ones(self.n_states, dtype=bool, device=self.device)
            mask[self.terminal_state] = False
            transition_function = transition_function[mask][:, :, mask]

        adjacency_matrix = (transition_function.sum(dim=1) > 0).float()
        degree_matrix = torch.diag(adjacency_matrix.sum(dim=1))

        if normalized:
            # Using pseudo-inverse for numerical stability
            sqrt_degree = torch.diag(
                1.0 / torch.sqrt(torch.clamp(degree_matrix.diag(), min=1e-10))
            )
            laplacian_matrix = torch.eye(
                adjacency_matrix.shape[0], device=self.device
            ) - torch.matmul(sqrt_degree, torch.matmul(adjacency_matrix, sqrt_degree))
        else:
            laplacian_matrix = degree_matrix - adjacency_matrix

        return laplacian_matrix


class RewardModel:
    def __init__(
        self,
        n_actions: int,
        n_states: int | None = None,
        device: Optional[str] = None,
    ):
        self.device = device or torch.device("cpu")
        assert n_actions is not None

        # if n_states is None, set it to 1 for the terminal state
        n_states = n_states if n_states is not None else 1
        self.reward_counts = torch.zeros((n_states, n_actions), device=self.device)
        self.reward_sums = torch.zeros((n_states, n_actions), device=self.device)

        # Initialize terminal state
        self.reward_counts[-1, :] = 1
        self.reward_sums[-1, :] = 0  # Zero reward for terminal state

    def reset(self, n_states: int | None) -> None:
        """Reset the reward model."""
        # initialize with terminal state if no states are provided
        n_states = n_states if n_states is not None else 1

        self.reward_counts = torch.zeros((n_states, self.n_actions), device=self.device)
        self.reward_sums = torch.zeros((n_states, self.n_actions), device=self.device)

        # Initialize terminal state
        self.reward_counts[-1, :] = 1
        self.reward_sums[-1, :] = 0

    @property
    def n_states(self) -> int:
        return self.reward_counts.shape[0]

    @property
    def n_actions(self) -> int:
        return self.reward_counts.shape[1]

    @property
    def terminal_state(self) -> int:
        return self.reward_counts.shape[0] - 1

    def update(self, state: int, action: int, reward: float) -> None:
        """Update the reward model with a new observation."""
        if not (0 <= state < self.n_states and 0 <= action < self.n_actions):
            raise ValueError(f"Invalid state {state} or action {action}")

        self.reward_counts[state, action] += 1
        self.reward_sums[state, action] += reward

    def get_reward_function(self) -> Tensor:
        """Get the current reward function estimates."""
        return torch.nan_to_num(
            self.reward_sums / (self.reward_counts + 1e-10), nan=0.0
        )

    def check_add_new_state(self, state: int) -> None:
        """Add a new state to the reward model."""
        if state < self.terminal_state:
            return

        new_counts = torch.zeros(
            (self.n_states + 1, self.n_actions), device=self.device
        )
        new_sums = torch.zeros((self.n_states + 1, self.n_actions), device=self.device)

        # Copy existing rewards except terminal state
        new_counts[:-2, :] = self.reward_counts[:-1, :]
        new_sums[:-2, :] = self.reward_sums[:-1, :]

        # Initialize new terminal state
        new_counts[-1, :] = 1
        new_sums[-1, :] = 0  # Zero reward for terminal state

        self.reward_counts = new_counts
        self.reward_sums = new_sums


class ModelBasedAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.95,
        device: Optional[str] = None,
        iterations: int = 500,
        epsilon: float = 1e-6,
    ):
        """Initialize the model-based RL agent.

        Args:
            n_states: Number of states in the MDP
            n_actions: Number of actions in the MDP
            gamma: Discount factor
            device: Device to use for computations
            iterations: Maximum number of iterations for value iteration
            terminal_state: Terminal state index (if any)
            epsilon: Convergence threshold for value iteration
        """
        n_states = n_states + 1  # Add terminal state
        self.device = device or torch.device("cpu")

        self.transitions = TransitionModel(n_actions, n_states, self.device)
        self.rewards = RewardModel(n_actions, n_states, self.device)

        self.gamma = gamma
        self.iterations = iterations
        self.epsilon = epsilon

        # Inlclude terminal state in Q-values and value function
        self.q_values = torch.zeros((n_states, n_actions), device=self.device)
        self.value_function = torch.zeros(n_states, device=self.device)

    def reset(self, n_states: int | None = None) -> None:
        """Reset the agent's models."""
        self.transitions.reset(n_states)
        self.rewards.reset(n_states)

        self.q_values = torch.zeros((self.n_states, self.n_actions), device=self.device)
        self.value_function = torch.zeros(self.n_states, device=self.device)

    @property
    def n_states(self) -> int:
        return self.transitions.n_states

    @property
    def n_actions(self) -> int:
        return self.transitions.n_actions

    @property
    def terminal_state(self) -> int:
        return self.transitions.terminal_state

    def check_add_new_state(self, state: int) -> None:
        # update the value function
        # adjust for terminal state, which is never visited
        assert isinstance(state, int)
        if state >= (self.n_states - 1):
            v = torch.zeros(self.n_states + 1, device=self.device)
            v[: self.n_states - 1] = self.value_function[: self.n_states - 1]

            q = torch.zeros(self.n_states + 1, self.n_actions, device=self.device)
            q[: self.n_states - 1] = self.q_values[: self.n_states - 1]

            self.value_function = v
            self.q_values = q

        # update the transition and reward models
        self.transitions.check_add_new_state(state)
        self.rewards.check_add_new_state(state)

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """Update the agent's models with a new transition.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        assert isinstance(state, int), f"Expected state as int, was {type(state)}"
        assert isinstance(action, int), f"Expected action as int, was {type(action)}"
        assert isinstance(
            reward, (float, int)
        ), f"Expected reward as float or int, was {type(reward)}"
        assert isinstance(
            next_state, int
        ), f"Expected next_state as int, was {type(next_state)}"
        assert isinstance(done, bool), f"Expected done as bool, was {type(done)}"

        # expand the state space if necessary for both the state and successor state if necessary
        self.check_add_new_state(state)
        self.check_add_new_state(next_state)

        self.transitions.update(state, action, next_state, done)
        self.rewards.update(state, action, reward)

    def estimate(self) -> None:
        """Estimate the optimal value function and policy."""
        transition_function = self.transitions.get_transition_function()
        reward_function = self.rewards.get_reward_function()
        self.q_values, self.value_function = value_iteration(
            transition_function,
            reward_function,
            gamma=self.gamma,
            epsilon=self.epsilon,
            max_iterations=self.iterations,
        )

    def estimate_value_function(self, return_terminal_state: bool = False) -> Tensor:
        """Estimate the optimal value function using value iteration.

        Args:
            return_terminal_state: Whether to include the terminal state in the output

        Returns:
            Tensor: Optimal value function
        """
        self.estimate()

        if not return_terminal_state:
            mask = torch.ones(self.n_states, dtype=bool, device=self.device)
            mask[self.terminal_state] = False
            value_function = self.value_function[mask]

        return value_function

    def get_q_values(self, state: int | ArrayLike) -> Tensor:
        # Note: this is a get function, so it should not update the state-space
        state = maybe_convert_to_list(state)
        if isinstance(state, list):
            return torch.stack([self.get_q_values(s) for s in state])

        # adjust for terminal state, which is never visited
        if state >= (self.n_states - 1):
            return torch.zeros(self.n_actions, device=self.device)
        return self.q_values[state]

    def get_policy(self, deterministic: bool = True, temperature: int = 1) -> Tensor:
        """Get the current policy of the agent.

        Args:
            deterministic: Whether to return a deterministic policy

        Returns:
            Tensor: Policy matrix of shape (n_states, n_actions) containing probabilities
                   or a one-hot vector for deterministic policies
        """
        self.estimate()

        if deterministic:
            return torch.eye(self.n_actions, device=self.device)[
                self.q_values.argmax(dim=1)
            ]
        else:
            # Boltzmann policy
            policy = torch.softmax(self.q_values / temperature, dim=1)
            return policy

    def get_graph_laplacian(
        self, normalized: bool = True, return_terminal_state: bool = False
    ) -> Tensor:
        """Get the graph Laplacian of the transition model.

        Args:
            normalized: Whether to return the normalized Laplacian
            return_terminal_state: Whether to include the terminal state

        Returns:
            Tensor: Graph Laplacian matrix
        """
        return self.transitions.estimate_graph_laplacian(
            normalized, return_terminal_state
        )


class DynaWithViAgent(ModelBasedAgent):
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.95,
        device: Optional[str] = None,
        iterations: int = 500,
        epsilon: float = 0.000001,
        learning_rate: float = 0.1,
        dyna_updates: int = 10,  # Number of Dyna planning updates per real update
    ):
        super().__init__(n_states, n_actions, gamma, device, iterations, epsilon)
        self.learning_rate = learning_rate
        self.dyna_updates = dyna_updates

    def _td_update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """Helper method for TD update of Q-values."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * torch.max(self.q_values[next_state])

        td_error = target - self.q_values[state, action]
        self.q_values[state, action] += self.learning_rate * td_error

    def _dyna_planning(self) -> None:
        """Perform Dyna planning updates using the learned model."""
        # Get current transition and reward models
        transition_probs = self.transitions.get_transition_function()
        rewards = self.rewards.get_reward_function()

        # Only consider states and actions that have been visited
        valid_states = torch.where(self.transitions.state_action_visited.any(dim=1))[0]

        for _ in range(self.dyna_updates):
            # Sample a random state-action pair that we've seen before
            state_idx = torch.randint(0, len(valid_states), (1,))
            state = valid_states[state_idx]

            # Only sample from actions we've tried in this state
            valid_actions = torch.where(self.transitions.state_action_visited[state])[0]
            action_idx = torch.randint(0, len(valid_actions), (1,))
            action = valid_actions[action_idx]

            # Sample next state from model
            next_state_probs = transition_probs[state, action]
            next_state = torch.multinomial(next_state_probs, 1).item()

            # Get reward from model
            reward = rewards[state, action].item()

            # Done if we transition to terminal state
            done = next_state == self.terminal_state

            # Perform TD update using imagined experience
            self._td_update(state, action, reward, next_state, done)

    def update(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        # Update the model (transition and reward models)
        super().update(state, action, reward, next_state, done)

        # Real experience TD update
        self._td_update(state, action, reward, next_state, done)

        # # Dyna planning updates
        # self._dyna_planning()

    def log_pmf(self, state: int, temperature: float = 1.0) -> Tensor:
        """Get the log-probabilities of each action in a state."""
        q_values = self.get_q_values(state)
        return torch.nn.functional.log_softmax(q_values / temperature, dim=0)

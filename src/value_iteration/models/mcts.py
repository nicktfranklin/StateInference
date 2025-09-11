from abc import ABC, abstractmethod
from collections import namedtuple
from random import choice
from typing import List, Tuple, Union

import numpy as np
from scipy import sparse
from tqdm import tqdm

from value_iteration.models.utils import (
    calculate_sr_from_transitions,
    inverse_cmf_sampler,
    softmax,
)

N_ACTIONS = 4


class _Node(ABC):
    @abstractmethod
    def find_random_child(self) -> int:
        return 0

    @abstractmethod
    def expand(self) -> int:
        return 0

    @abstractmethod
    def __hash__(self):
        return 123456789

    @abstractmethod
    def __eq__(self, other):
        return True


_GWN = namedtuple("GridWorldTuple", "state n_actions")


class GridWorldNode(ABC, _GWN):
    def find_random_child(self) -> int:
        # choose a random action
        return choice(np.arange(0, self.n_actions))

    def expand(self) -> int:
        return self.find_random_child()


class MCTS:
    """Monte Carlo Tree Search implementation for grid world environments.

    This class implements the standard MCTS algorithm with Upper Confidence Bounds (UCB)
    for action selection. The algorithm builds a search tree by iteratively selecting
    actions, expanding nodes, simulating rollouts, and backpropagating rewards.

    The MCTS algorithm follows these four main phases:
    1. Selection: Navigate from root using UCB until reaching unexplored node
    2. Expansion: Add new child node to the tree
    3. Simulation: Perform random rollout from new node
    4. Backpropagation: Update Q-values and visit counts along path
    """

    def __init__(
        self,
        end_states: List[int],
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        state_reward_function: np.ndarray,
        exploration_weight: float = 2**0.5,
        n_actions: int = 4,
        epsilon: float = 1,  # minimum visitiation constant
        max_depth: int = 1000,
        n_sims: int = 1,
        gamma: float = 0.99,
    ):
        """Initialize MCTS algorithm.

        Args:
            end_states: List of terminal state indices
            transition_functions: List of transition matrices, one per action
            state_reward_function: Array of rewards for each state
            exploration_weight: UCB exploration parameter (default: sqrt(2))
            n_actions: Number of available actions (default: 4)
            epsilon: Minimum visit count for initialization (default: 1)
            max_depth: Maximum simulation depth (default: 1000)
            n_sims: Number of simulations per rollout (default: 1)
            gamma: Discount factor for rewards (default: 0.99)
        """
        self.end_states = end_states
        self.transition_functions = transition_functions
        self.state_reward_function = state_reward_function
        self.exploration_weight = exploration_weight
        self.n_states = len(state_reward_function)
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.expanded_nodes = dict()
        self.n_sims = n_sims
        self.gamma = gamma
        self.bellman_updates = 0

        self.q = dict()
        self.n = dict()

        # convert the sparse transition matricies to np.ndarrays
        self.transition_functions: List[np.ndarray] = [
            t for t in transition_functions if type(t) == np.ndarray
        ] + [t.toarray() for t in transition_functions if type(t) == sparse.csr_matrix]

    def _sample_sucessor_state(self, state: int, action: int) -> int:
        t_a = self.transition_functions[action][state, :]
        sucessor_state = inverse_cmf_sampler(t_a)
        return sucessor_state

    def _sample_sucessor_node(self, state: int, action: int) -> GridWorldNode:
        sucessor_state = self._sample_sucessor_state(state, action)
        if sucessor_state in self.expanded_nodes:
            return self.expanded_nodes[sucessor_state]
        return GridWorldNode(state=sucessor_state, n_actions=self.n_actions)

    def _is_unexplored(self, node: GridWorldNode) -> bool:
        return node.state not in self.expanded_nodes

    def _draw_random_sucessor(self, node: GridWorldNode, action: int):
        # sample a sucessor state
        t_a = self.transition_functions[action][node.state, :]
        sucessor_state = inverse_cmf_sampler(t_a)

        return GridWorldNode(state=sucessor_state, n_actions=self.n_actions)

    def _get_reward(self, node: GridWorldNode) -> bool:
        return self.state_reward_function[node.state]

    def _is_expanded(self, node: GridWorldNode) -> bool:
        return node in self.q.keys()

    def _is_terminal(self, node: GridWorldNode) -> bool:
        return node.state in self.end_states

    def _update_child_value(
        self, node: GridWorldNode, child_action: int, reward: float
    ) -> None:
        if self._is_terminal(node):
            return

        self.q[node.state][child_action] += reward
        self.n[node.state][child_action] += 1
        self.bellman_updates += 1

    def _get_ucb_values(self, node: GridWorldNode) -> np.ndarray:
        r = self.q[node.state] / self.n[node.state]
        ucb = np.sqrt(2 * np.log(self.n[node.state].sum()) / self.n[node.state])
        return r + self.exploration_weight * ucb

    def _ucb_select_action(self, node: GridWorldNode) -> int:
        assert self.n[node.state].sum() >= 1, "Child Nodes have not been visited!"

        # deterministic UCB sampler
        return np.argmax(self._get_ucb_values(node))

    def _get_argmax_policy(self, node: GridWorldNode) -> int:
        return int(np.argmax(self.q[node.state] / self.n[node.state]))

    def _get_softmax_policy(self, node: GridWorldNode) -> np.ndarray:
        return softmax(self._get_ucb_values(node))

    def select(
        self,
        node: GridWorldNode,
    ) -> Tuple[List[Tuple[GridWorldNode, int]], GridWorldNode]:
        """Selection phase of MCTS: navigate tree using UCB until unexplored node.

        Traverses the tree from the given node using Upper Confidence Bounds (UCB)
        action selection until reaching either an unexplored node or a terminal state.

        Args:
            node: Starting node for tree traversal

        Returns:
            Tuple of (path, leaf_node) where path is list of (node, action) pairs
            taken during traversal and leaf_node is the final unexplored/terminal node
        """
        # path is list of (state, action) tuples
        path = []
        reward, depth = 0, 0

        while depth < self.max_depth:
            # print(depth)
            if self._is_unexplored(node) or self._is_terminal(node):
                return path, node
            # self.q[node.state] += self._get_reward(node)

            # draw an action, update the path
            action = self._ucb_select_action(node)
            path.append((node, action))

            # draw next state
            node = self._sample_sucessor_node(node.state, action)
            # reward += self._get_reward(node)
            depth += 1

        return path, node

    def expand(self, node: GridWorldNode) -> int:
        """Expansion phase of MCTS: add new node to the search tree.

        Adds the given node to the expanded_nodes dictionary and initializes
        its Q-values and visit counts. Returns the action selected for simulation.

        Args:
            node: Node to expand and add to the tree

        Returns:
            Action index selected for simulation, or -1 if node is terminal
        """
        if self._is_terminal(node):
            return -1
        self.expanded_nodes[node.state] = node

        self.q[node.state] = np.zeros(self.n_actions)
        self.n[node.state] = np.zeros(self.n_actions) + self.epsilon
        return node.expand()

    def simulate_state_value(self, start_node: GridWorldNode) -> List[float]:
        rewards = [0.0] * self.n_sims

        for ii in range(self.n_sims):
            node = start_node
            reward, depth = 0, 0

            while depth < self.max_depth:
                reward += self._get_reward(node) * (self.gamma**depth)
                action = node.find_random_child()
                node = self._draw_random_sucessor(node, action)
                depth += 1

            rewards[ii] = reward

        return rewards

    def _single_sim(self, node: GridWorldNode, action: int, return_path: bool = False):
        if return_path:
            path = [node]

        if self._is_terminal(node):
            return self._get_reward(node)

        node = self._draw_random_sucessor(node, action)
        reward = 0
        depth = 0
        while True:
            if return_path:
                path.append(node)
            reward += self._get_reward(node) * (self.gamma**depth)
            if self._is_terminal(node):
                if return_path:
                    return reward, path
                return reward
            action = node.find_random_child()
            node = self._draw_random_sucessor(node, action)
            depth += 1

    def simulate(self, node: GridWorldNode, action: int, k=None) -> float:
        """Simulation phase of MCTS: perform random rollouts to estimate value.

        Performs multiple random rollouts from the given node and action,
        returning the average discounted reward obtained.

        Args:
            node: Starting node for simulation
            action: Initial action to take
            k: Number of simulations (defaults to self.n_sims)

        Returns:
            Average discounted reward from simulations
        """
        iterations = k or self.n_sims
        return np.mean([self._single_sim(node, action) for _ in range(iterations)])

    def backpropagate(self, path: List[Tuple[GridWorldNode, int]], reward: float):
        """Backpropagation phase of MCTS: update Q-values along the search path.

        Updates Q-values and visit counts for all (node, action) pairs along
        the path from leaf back to root using the reward from simulation.

        Args:
            path: List of (node, action) pairs from selection phase
            reward: Reward value to backpropagate from simulation
        """
        # path is a sequence of (state, action) tuples
        for node, action in reversed(path):
            self._update_child_value(node, action, reward)

    def do_single_rollout(self, start_state):
        """Perform one complete MCTS iteration (select-expand-simulate-backpropagate).

        Executes all four phases of the MCTS algorithm starting from the given state:
        1. Selection: Navigate tree using UCB to find leaf node
        2. Expansion: Add leaf node to tree
        3. Simulation: Perform random rollout to estimate value
        4. Backpropagation: Update Q-values along path

        Args:
            start_state: GridWorldNode to start the rollout from

        Returns:
            Path taken during selection phase as list of (node, action) pairs
        """
        path, leaf = self.select(start_state)
        a = self.expand(leaf)
        reward = self.simulate(leaf, a)
        path.append((leaf, a))
        self.backpropagate(path, reward)
        return path

    def do_rollouts(self, start_state, k=10, progress_bar: bool = True):
        """Perform multiple MCTS rollouts to build the search tree.

        Repeatedly calls do_single_rollout to iteratively improve the search tree
        and value estimates. This is the main training loop for MCTS.

        Args:
            start_state: GridWorldNode to start all rollouts from
            k: Number of rollouts to perform (default: 10)
            progress_bar: Whether to show progress bar (default: True)
        """
        if progress_bar:
            _iterator = tqdm(range(k), desc="MCTS Rollouts")
        else:
            _iterator = range(k)
        for _ in _iterator:
            self.do_single_rollout(start_state)

    def get_policy(self):
        """Extract the learned policy from the MCTS tree.

        Returns the greedy policy based on Q-values for all expanded nodes.
        Non-expanded states get policy value -1 indicating unknown.

        Returns:
            numpy.ndarray: Policy array where pi[s] gives best action for state s,
            or -1 if state s has not been expanded
        """
        pi = np.ones_like(self.state_reward_function, dtype=int) * -1
        for state, node in self.expanded_nodes.items():
            pi[state] = self._get_argmax_policy(node)
        return pi

    def get_simulation_depth(self, node: GridWorldNode) -> int:
        if self._is_terminal(node):
            return 0

        depth = 0
        while True:
            depth += 1
            if self._is_terminal(node):
                return depth
            action = node.find_random_child()
            node = self._draw_random_sucessor(node, action)

    def get_selection_policy(self, beta: float) -> np.ndarray:
        policy = (
            np.ones((self.state_reward_function.shape[0], N_ACTIONS), dtype=float)
            / N_ACTIONS
        )
        for state, node in self.expanded_nodes.items():
            w = self._get_ucb_values(node)
            policy[state, :] = softmax(w, beta)
        return policy

    def get_node_visitations(self):
        visits = np.zeros_like(self.state_reward_function)
        for node in self.n.keys():
            visits[node] = np.sum(self.n[node]) - self.epsilon * self.n_actions
        return visits

    def simulate_policy(self, start_node: GridWorldNode):
        policy = self.get_policy()
        node = start_node
        path = [node]

        # use deterministic sampling, but randomly sample when
        # unknown
        ii = 0
        while not self._is_terminal(node):
            action = policy[node.state]
            if action == -1:
                action = choice(np.arange(N_ACTIONS))

            node = self._sample_sucessor_node(node.state, action)
            path.append(node.state)

            ii += 1
            if ii > 500:
                break

        return path

    def reset(self):
        self.expanded_nodes = dict()
        self.q = dict()
        self.n = dict()
        self.bellman_updates = 0

    def get_expanded_nodes(self) -> np.ndarray:
        states = np.zeros(self.n_states)
        for k in self.expanded_nodes.keys():
            states[k] = 1
        return states

    def batch_rollouts(
        self, start_node: GridWorldNode, n_batches: int, n_rollouts: int, beta: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        selection_pmf, visitations = [], []
        for _ in tqdm(range(n_batches), desc="Batch"):
            self.reset()
            self.do_rollouts(start_node, k=n_rollouts, progress_bar=False)
            selection_pmf.append(self.get_selection_policy(beta))
            visitations.append(self.get_expanded_nodes())

        # noinspection PyUnresolvedReferences
        return np.mean(selection_pmf, axis=0), np.mean(visitations, axis=0)


class MctsSr(MCTS):
    """MCTS with Successor Representation for improved value estimation.

    This variant of MCTS uses pre-computed state values from the Successor
    Representation to provide better heuristic estimates during simulation,
    potentially leading to faster convergence and better performance.

    The key difference from standard MCTS is that instead of random rollouts,
    it uses the SR-based state values for immediate value estimation.
    """

    def __init__(
        self,
        end_states: List[int],
        transition_functions: List[Union[np.ndarray, sparse.csr_matrix]],
        state_reward_function: np.ndarray,
        exploration_weight: float = 2**0.5,
        n_actions: int = 4,
        epsilon: float = 1,  # minimum visitiation constant
        max_depth: int = 1000,
        n_sims: int = 1,
        gamma: float = 0.99,
    ):
        """Initialize MCTS with Successor Representation.

        Extends the base MCTS with pre-computed state values using the
        Successor Representation under a random policy for better heuristics.

        Args:
            Same as MCTS.__init__. See parent class for parameter descriptions.
        """
        super().__init__(
            end_states,
            transition_functions,
            state_reward_function,
            exploration_weight,
            n_actions,
            epsilon,
            max_depth,
            n_sims,
            gamma,
        )

        # pre-compute state values under a random action policy, which we get by averaging the transition functions
        diffusion_transition_matrix = np.mean(self.transition_functions, axis=0)
        sr = calculate_sr_from_transitions(diffusion_transition_matrix, gamma=gamma)

        self.state_values = sr.dot(state_reward_function)

    def simulate(self, node: GridWorldNode, action: int, k=None) -> float:
        """Simulate using pre-computed SR state values instead of rollouts.

        Instead of performing random rollouts, this method uses the pre-computed
        state values from the Successor Representation to immediately return
        an estimate of the value after taking the given action.

        Args:
            node: Current node
            action: Action to take
            k: Unused (kept for interface compatibility)

        Returns:
            Pre-computed state value of the successor state
        """
        node = self._draw_random_sucessor(node, action)
        return self.state_values[node.state]

    def _single_sim(self, node: GridWorldNode, action: int, return_path: bool = False):
        pass

    def expand(self, node: GridWorldNode) -> int:
        if self._is_terminal(node):
            return -1
        self.expanded_nodes[node.state] = node

        self.q[node.state] = (
            np.zeros(self.n_actions) + np.max(self.state_values) * self.epsilon
        )
        self.n[node.state] = np.zeros(self.n_actions) + self.epsilon
        return node.expand()

    def update_heuristic_function(self, state_value_function: np.ndarray) -> None:
        """Update the heuristic state value function.

        Allows updating the state values used for simulation with new estimates,
        useful for iterative improvement or when better value estimates become available.

        Args:
            state_value_function: New array of state values to use for heuristics
        """
        self.state_values = state_value_function

    def expected_discounted_rewards(
        self, start_node: GridWorldNode, n_simulations: int = 100
    ) -> float:
        """Calculate expected discounted future rewards from a given start node.

        Performs multiple random rollouts from the start node to estimate the expected
        discounted cumulative reward under the current policy or random exploration.

        Args:
            start_node: Node to start the reward calculation from
            n_simulations: Number of simulation runs to average over (default: 1000)

        Returns:
            Expected discounted future reward as a float
        """
        total_rewards = []

        for _ in range(n_simulations):
            node = start_node
            total_reward = 0
            depth = 0

            while depth < self.max_depth and not self._is_terminal(node):
                # Get immediate reward
                immediate_reward = self._get_reward(node)
                total_reward += immediate_reward * (self.gamma**depth)

                # Select action (use learned policy if available, otherwise random)
                if node.state in self.expanded_nodes:
                    action = self._get_argmax_policy(node)
                else:
                    action = node.find_random_child()

                # Move to next state
                node = self._draw_random_sucessor(node, action)
                depth += 1

            # Add terminal reward if reached
            if self._is_terminal(node):
                terminal_reward = self._get_reward(node)
                total_reward += terminal_reward * (self.gamma**depth)

            total_rewards.append(total_reward)

        return np.mean(total_rewards), np.std(total_rewards)

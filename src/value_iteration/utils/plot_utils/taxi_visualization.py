"""Visualization functions for the counter-based taxi problem.

This module provides visualization tools for the taxi gridworld problem
that uses a counter-based state representation (row, col, counter).
"""

from typing import List, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.sparse import csr_matrix

# Import the counter-based taxi problem implementation
from value_iteration.task.taxi_problem import (
    TaxiGridWorld,
    create_taxi_transition_functions,
    get_valid_neighbors,
    make_gridworld_taxi,
    pos_to_state,
    state_to_pos,
)


class TaxiVisualization:
    """Visualization class for counter-based taxi problem."""

    def __init__(
        self,
        grid_size: int = 10,
        agent_pos: Tuple[int, int] = (0, 0),
        stop1_pos: Tuple[int, int] = (4, 5),
        stop2_pos: Tuple[int, int] = (7, 2),
        goal_pos: Tuple[int, int] = (9, 9),
    ):
        self.grid_size = grid_size
        self.agent_pos = agent_pos
        self.stop1_pos = stop1_pos
        self.stop2_pos = stop2_pos
        self.goal_pos = goal_pos
        self.total_states = 3 * grid_size * grid_size

        # Colors and styling
        self.colors = {
            "agent": "#4CAF50",  # Green
            "stop1": "#FF9800",  # Orange
            "stop2": "#9C27B0",  # Purple
            "goal": "#F44336",  # Red
            "grid": "#E0E0E0",  # Light gray
        }

        # Create the taxi problem using our counter-based implementation
        self.transitions, self.rewards, self.policy = make_gridworld_taxi(
            grid_size=grid_size,
            agent_pos=agent_pos,
            stop1_pos=stop1_pos,
            stop2_pos=stop2_pos,
            goal_pos=goal_pos,
        )

    def visualize_gridworld(self, figsize: Tuple[int, int] = (8, 8)) -> Tuple:
        """Visualize the taxi gridworld with counter-based stops."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Create the grid
        for i in range(self.grid_size + 1):
            ax.axhline(i, color=self.colors["grid"], linewidth=1)
            ax.axvline(i, color=self.colors["grid"], linewidth=1)

        # Add agent (starting position)
        agent_rect = Rectangle(
            (self.agent_pos[1], self.grid_size - 1 - self.agent_pos[0]),
            1,
            1,
            facecolor=self.colors["agent"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(agent_rect)
        ax.text(
            self.agent_pos[1] + 0.5,
            self.grid_size - 1 - self.agent_pos[0] + 0.5,
            "🤖",
            fontsize=20,
            ha="center",
            va="center",
        )

        # Add first stop
        stop1_rect = Rectangle(
            (self.stop1_pos[1], self.grid_size - 1 - self.stop1_pos[0]),
            1,
            1,
            facecolor=self.colors["stop1"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(stop1_rect)
        ax.text(
            self.stop1_pos[1] + 0.5,
            self.grid_size - 1 - self.stop1_pos[0] + 0.5,
            "⭐",
            fontsize=20,
            ha="center",
            va="center",
        )

        # Add second stop
        stop2_rect = Rectangle(
            (self.stop2_pos[1], self.grid_size - 1 - self.stop2_pos[0]),
            1,
            1,
            facecolor=self.colors["stop2"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(stop2_rect)
        ax.text(
            self.stop2_pos[1] + 0.5,
            self.grid_size - 1 - self.stop2_pos[0] + 0.5,
            "💎",
            fontsize=20,
            ha="center",
            va="center",
        )

        # Add goal
        goal_rect = Rectangle(
            (self.goal_pos[1], self.grid_size - 1 - self.goal_pos[0]),
            1,
            1,
            facecolor=self.colors["goal"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(goal_rect)
        ax.text(
            self.goal_pos[1] + 0.5,
            self.grid_size - 1 - self.goal_pos[0] + 0.5,
            "🎯",
            fontsize=20,
            ha="center",
            va="center",
        )

        # Set up the plot
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.set_title(
            "Taxi Problem with Counter-Based Stops", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Set ticks
        ax.set_xticks(range(0, self.grid_size + 1, 2))
        ax.set_yticks(range(0, self.grid_size + 1, 2))

        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors["agent"], label="Taxi (Start)"),
            patches.Patch(color=self.colors["stop1"], label="Stop 1 (Counter: 0→1)"),
            patches.Patch(color=self.colors["stop2"], label="Stop 2 (Counter: 1→2)"),
            patches.Patch(color=self.colors["goal"], label="Goal (Counter=2)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        return fig, ax

    def create_adjacency_matrix(self) -> Tuple[csr_matrix, dict]:
        """Create adjacency matrix from transition functions."""
        # Sum all action transition matrices to get overall connectivity
        if isinstance(self.transitions[0], csr_matrix):
            adj_matrix = sum(self.transitions)
        else:
            adj_matrix = csr_matrix(sum(self.transitions))

        # Create state info dictionary
        state_info = {
            "agent_state": pos_to_state(
                self.agent_pos[0], self.agent_pos[1], 0, self.grid_size
            ),
            "stop1_counter0": pos_to_state(
                self.stop1_pos[0], self.stop1_pos[1], 0, self.grid_size
            ),
            "stop1_counter1": pos_to_state(
                self.stop1_pos[0], self.stop1_pos[1], 1, self.grid_size
            ),
            "stop2_counter1": pos_to_state(
                self.stop2_pos[0], self.stop2_pos[1], 1, self.grid_size
            ),
            "stop2_counter2": pos_to_state(
                self.stop2_pos[0], self.stop2_pos[1], 2, self.grid_size
            ),
            "goal_counter2": pos_to_state(
                self.goal_pos[0], self.goal_pos[1], 2, self.grid_size
            ),
            "total_states": self.total_states,
            "grid_size": self.grid_size,
            "num_layers": 3,
        }

        return adj_matrix, state_info

    def visualize_adjacency_matrix(
        self, max_states_to_show: int = 50, show_self_transitions: bool = True
    ) -> None:
        """Visualize a subset of the adjacency matrix.

        Args:
            max_states_to_show: Number of states to show in each dimension
            show_self_transitions: If False, mask self-transitions (diagonal) for cleaner view
        """
        adj_matrix, _ = self.create_adjacency_matrix()

        # Show only first max_states_to_show x max_states_to_show for readability
        if hasattr(adj_matrix, "toarray"):
            subset = adj_matrix[:max_states_to_show, :max_states_to_show].toarray()
        else:
            subset = adj_matrix[:max_states_to_show, :max_states_to_show].copy()

        # Optionally mask self-transitions
        if not show_self_transitions:
            np.fill_diagonal(subset, 0)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            subset, cmap="Blues", cbar=True, square=True, xticklabels=5, yticklabels=5
        )

        title = f"Transition Matrix (First {max_states_to_show}x{max_states_to_show} states)"
        if not show_self_transitions:
            title += " - Self-transitions Hidden"
        plt.title(title)
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.show()

    def visualize_graph_only(
        self, figsize: Tuple[int, int] = (10, 8), show_self_transitions: bool = False
    ) -> Tuple:
        """Visualize just the state transition graph.

        Args:
            figsize: Figure size as (width, height)
            show_self_transitions: If False, mask self-transitions for cleaner visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        adj_matrix, state_info = self.create_adjacency_matrix()
        self._draw_graph_on_axis(ax, adj_matrix, show_self_transitions)

        title = "State Transition Graph"
        if not show_self_transitions:
            title += " (Self-transitions Hidden)"
        ax.set_title(title, fontsize=16, fontweight="bold")

        # Add info text
        info_text = (
            f"Nodes: {state_info['total_states']} states\n"
            f"Layers: {state_info['num_layers']} (counter=0,1,2)\n"
            f"Grid: {state_info['grid_size']}×{state_info['grid_size']}"
        )

        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()
        return fig, ax

    def visualize_combined_view(
        self, figsize: Tuple[int, int] = (15, 6), show_self_transitions: bool = False
    ) -> Tuple:
        """Create combined visualization with gridworld and graph representation.

        Args:
            figsize: Figure size as (width, height)
            show_self_transitions: If False, mask self-transitions in graph for cleaner visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left panel: Gridworld visualization
        self._draw_gridworld_on_axis(ax1)
        ax1.set_title("Counter-Based Taxi Problem", fontsize=14, fontweight="bold")
        ax1.text(
            -0.1, 1.02, "A", transform=ax1.transAxes, fontsize=24, fontweight="bold"
        )

        # Right panel: Graph representation
        adj_matrix, _ = self.create_adjacency_matrix()
        self._draw_graph_on_axis(ax2, adj_matrix, show_self_transitions)

        title = "State Transition Graph"

        ax2.set_title(title, fontsize=14, fontweight="bold")
        ax2.text(
            -0.1, 1.02, "B", transform=ax2.transAxes, fontsize=24, fontweight="bold"
        )

        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        return fig, (ax1, ax2)

    def _draw_gridworld_on_axis(self, ax):
        """Helper method to draw gridworld on given axis."""
        # Create the grid
        for i in range(self.grid_size + 1):
            ax.axhline(i, color=self.colors["grid"], linewidth=1)
            ax.axvline(i, color=self.colors["grid"], linewidth=1)

        # Add agent
        agent_rect = Rectangle(
            (self.agent_pos[1], self.grid_size - 1 - self.agent_pos[0]),
            1,
            1,
            facecolor=self.colors["agent"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(agent_rect)
        ax.text(
            self.agent_pos[1] + 0.5,
            self.grid_size - 1 - self.agent_pos[0] + 0.5,
            "🤖",
            fontsize=20,
            ha="center",
            va="center",
        )

        # Add first stop
        stop1_rect = Rectangle(
            (self.stop1_pos[1], self.grid_size - 1 - self.stop1_pos[0]),
            1,
            1,
            facecolor=self.colors["stop1"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(stop1_rect)
        ax.text(
            self.stop1_pos[1] + 0.5,
            self.grid_size - 1 - self.stop1_pos[0] + 0.5,
            "⭐",
            fontsize=20,
            ha="center",
            va="center",
        )

        # Add second stop
        stop2_rect = Rectangle(
            (self.stop2_pos[1], self.grid_size - 1 - self.stop2_pos[0]),
            1,
            1,
            facecolor=self.colors["stop2"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(stop2_rect)
        ax.text(
            self.stop2_pos[1] + 0.5,
            self.grid_size - 1 - self.stop2_pos[0] + 0.5,
            "💎",
            fontsize=20,
            ha="center",
            va="center",
        )

        # Add goal
        goal_rect = Rectangle(
            (self.goal_pos[1], self.grid_size - 1 - self.goal_pos[0]),
            1,
            1,
            facecolor=self.colors["goal"],
            alpha=0.7,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(goal_rect)
        ax.text(
            self.goal_pos[1] + 0.5,
            self.grid_size - 1 - self.goal_pos[0] + 0.5,
            "🎯",
            fontsize=20,
            ha="center",
            va="center",
        )

        # Set up the axis
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_xticks(range(0, self.grid_size + 1, 2))
        ax.set_yticks(range(0, self.grid_size + 1, 2))

        # Add legend
        legend_elements = [
            patches.Patch(color=self.colors["agent"], label="Taxi"),
            patches.Patch(color=self.colors["stop1"], label="Stop 1"),
            patches.Patch(color=self.colors["stop2"], label="Stop 2"),
            patches.Patch(color=self.colors["goal"], label="Goal"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    def _draw_graph_on_axis(self, ax, adj_matrix, show_self_transitions=False):
        """Helper method to draw graph representation on given axis."""
        # Convert to dense for NetworkX
        if hasattr(adj_matrix, "toarray"):
            adj_dense = adj_matrix.toarray()
        else:
            adj_dense = adj_matrix.copy()

        # Optionally mask self-transitions for cleaner visualization
        if not show_self_transitions:
            np.fill_diagonal(adj_dense, 0)

        # Create graph
        G = nx.from_numpy_array(adj_dense)
        pos = nx.spring_layout(G, seed=42, k=0.1, iterations=1000)

        # Draw graph
        nx.draw(G, pos, with_labels=False, node_color="lightblue", node_size=20, ax=ax)

    def analyze_connectivity(self) -> dict:
        """Analyze the connectivity properties of the counter-based system."""
        adj_matrix, state_info = self.create_adjacency_matrix()

        print("=== Counter-Based Taxi Problem Analysis ===")
        print(f"Total states: {self.total_states}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Number of layers (counter values): 3")
        print(f"Total connections: {adj_matrix.nnz}")
        print(
            f"Average connections per state: {adj_matrix.nnz / self.total_states:.2f}"
        )
        print(f"Matrix density: {adj_matrix.nnz / (self.total_states ** 2) * 100:.4f}%")

        print(f"\nKey States:")
        print(f"Agent starting state (0,0,counter=0): {state_info['agent_state']}")
        print(f"Stop 1 at counter=0: {state_info['stop1_counter0']}")
        print(f"Stop 1 at counter=1: {state_info['stop1_counter1']}")
        print(f"Stop 2 at counter=1: {state_info['stop2_counter1']}")
        print(f"Stop 2 at counter=2: {state_info['stop2_counter2']}")
        print(f"Goal at counter=2: {state_info['goal_counter2']}")

        # Check goal state absorption
        goal_state = state_info["goal_counter2"]
        goal_connections = adj_matrix[goal_state, :].nonzero()[1]
        is_absorbing = len(goal_connections) == 1 and goal_connections[0] == goal_state

        print(f"\nGoal State Analysis:")
        print(f"Goal state {goal_state} is absorbing: {is_absorbing}")
        print(f"Goal connections: {goal_connections}")

        return {
            "total_states": self.total_states,
            "total_connections": adj_matrix.nnz,
            "density": adj_matrix.nnz / (self.total_states**2),
            "goal_is_absorbing": is_absorbing,
            "state_info": state_info,
        }


def create_taxi_visualization(
    grid_size: int = 10,
    agent_pos: Tuple[int, int] = (0, 0),
    stop1_pos: Tuple[int, int] = (4, 5),
    stop2_pos: Tuple[int, int] = (7, 2),
    goal_pos: Tuple[int, int] = (9, 9),
) -> TaxiVisualization:
    """Convenience function to create a TaxiVisualization instance."""
    return TaxiVisualization(grid_size, agent_pos, stop1_pos, stop2_pos, goal_pos)

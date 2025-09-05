#!/usr/bin/env python3
"""
MCTS Analysis Script

This script replicates the analysis from MCTS.ipynb, generating the same plots
and saving them to files. It also generates an HTML report with all results.
"""

import base64
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Set up matplotlib for non-interactive use
plt.ioff()

import value_iteration.task.thread_the_needle as e

# Import project modules
from value_iteration.models import MCTS, GridWorldNode, MctsSr, ValueIterationNetwork
from value_iteration.models.sr import get_optimal_sr_from_transitions
from value_iteration.task.thread_the_needle import find_sortest_path_length


def save_plot(fig, filename, dpi=300):
    """Save plot to images directory"""
    os.makedirs("images", exist_ok=True)
    filepath = os.path.join("images", filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return filepath


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding"""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return image_base64


class HTMLReport:
    """Generate HTML report with plots and analysis"""

    def __init__(self):
        self.html_content = []
        self.add_header()

    def add_header(self):
        self.html_content.append(
            """
<!DOCTYPE html>
<html>
<head>
    <title>MCTS Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .plot { margin: 20px 0; text-align: center; }
        .plot img { max-width: 100%; height: auto; }
        h1, h2 { color: #333; }
        .section { margin: 40px 0; }
        .description { margin: 20px 0; font-style: italic; color: #666; }
    </style>
</head>
<body>
    <h1>MCTS Analysis Report</h1>
    <p>This report replicates the analysis from MCTS.ipynb notebook.</p>
"""
        )

    def add_section(self, title, description=""):
        self.html_content.append(
            f"""
    <div class="section">
        <h2>{title}</h2>
        {f'<p class="description">{description}</p>' if description else ''}
"""
        )

    def add_plot(self, fig, title="", description=""):
        image_base64 = fig_to_base64(fig)
        self.html_content.append(
            f"""
        <div class="plot">
            {f'<h3>{title}</h3>' if title else ''}
            <img src="data:image/png;base64,{image_base64}" alt="{title}">
            {f'<p>{description}</p>' if description else ''}
        </div>
"""
        )

    def close_section(self):
        self.html_content.append("    </div>")

    def save(self, filename="mcts_analysis_report.html"):
        self.html_content.append(
            """
</body>
</html>
"""
        )
        with open(filename, "w") as f:
            f.write("\n".join(self.html_content))


def main():
    print("Starting MCTS analysis...")

    # Initialize HTML report
    report = HTMLReport()

    # Environment setup
    n_rows, n_columns = 20, 20
    gamma = 0.95

    print("Setting up environment...")
    transition_functions, reward_functions, optimal_policy = e.make_thread_the_needle(
        n_rows=n_rows,
        n_columns=n_columns,
        movement_penalty=0.1,
        slip_probability=0.001,
        random_movement_on_error=True,
        sparse=False,
    )

    M = get_optimal_sr_from_transitions(
        transition_functions, optimal_policy, gamma=gamma
    )

    transition_functions_diffusion = [
        np.mean(transition_functions, axis=0) for _ in range(4)
    ]

    M_diff = get_optimal_sr_from_transitions(
        transition_functions_diffusion, optimal_policy, gamma=gamma
    )

    v_optimal = M.dot(reward_functions)
    v_diffus = M_diff.dot(reward_functions)

    # Plot 1: State-value functions comparison
    report.add_section(
        "State-Value Function Comparison",
        "Comparison between optimal and diffusion policy state-value functions",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(v_optimal.reshape(n_rows, n_columns))
    axes[0].set_title("Optimal State-Value Function")
    axes[1].imshow(v_diffus.reshape(n_rows, n_columns))
    axes[1].set_title("Diffusion State-Value Function")

    save_plot(fig, "state_value_comparison.png")
    report.add_plot(
        fig, "State-Value Functions", "Left: Optimal policy, Right: Diffusion policy"
    )

    # Calculate distances to goal
    print("Calculating distances to goal...")
    f = lambda s: find_sortest_path_length(
        v_optimal, transition_functions, 0, s, n_rows, n_columns
    )
    distances_to_goal = np.array([f(s) for s in range(n_rows * n_columns)])

    # Plot 2: Distance to goal visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(distances_to_goal.reshape(n_rows, n_columns))
    ax.set_title("Distance to Goal")

    save_plot(fig, "distance_to_goal.png")
    report.add_plot(
        fig, "Distance to Goal", "Shortest path distance from each state to the goal"
    )

    # Plot 3: Heuristic function against distance
    report.add_section(
        "Heuristic Analysis",
        "Analysis of how state values correlate with distance to goal",
    )

    df = pd.DataFrame(
        {
            "Distance to goal": np.concatenate([distances_to_goal] * 2),
            "State-Value": np.concatenate([v_optimal, v_diffus]),
            "Policy": ["Optimal"] * (n_rows * n_columns)
            + ["Diffusion"] * (n_rows * n_columns),
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x="Distance to goal", y="State-Value", hue="Policy", data=df, ax=ax)

    save_plot(fig, "heurisitic_function_againt_distance_to_goal.png")
    report.add_plot(
        fig,
        "State-Value vs Distance to Goal",
        "How state values change with distance to goal for different policies",
    )

    # Plot 4: State-value gradients
    def get_grad(vec: np.ndarray) -> np.ndarray:
        return vec[:-1] - vec[1:]

    df2 = pd.DataFrame(
        {
            "Distance to goal": np.concatenate(
                [
                    df.groupby(["Policy", "Distance to goal"])
                    .mean()
                    .loc["Optimal"]
                    .index[1:]
                ]
                * 2
            ),
            "State-Value Gradient": np.concatenate(
                [
                    get_grad(
                        df.groupby(["Policy", "Distance to goal"])
                        .mean()
                        .loc["Optimal"]
                        .values
                    ),
                    get_grad(
                        df.groupby(["Policy", "Distance to goal"])
                        .mean()
                        .loc["Diffusion"]
                        .values
                    ),
                ]
            ).reshape(-1),
            "Policy": ["Optimal"] * (df["Distance to goal"].max())
            + ["Diffusion"] * (df["Distance to goal"].max()),
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x="Distance to goal", y="State-Value Gradient", hue="Policy", data=df2, ax=ax
    )

    save_plot(fig, "state_value_gradients.png")
    report.add_plot(
        fig,
        "State-Value Gradients",
        "Gradients of state values with respect to distance",
    )

    # MCTS Variance Analysis
    report.add_section(
        "MCTS Variance Analysis",
        "Analysis of variance in MCTS value estimates through simulation",
    )

    print("Running MCTS variance analysis...")
    goals_state = 0
    mcts_kwargs = dict(
        end_states=[goals_state],
        transition_functions=transition_functions,
        state_reward_function=reward_functions,
        exploration_weight=2**0.5,
        max_depth=500,
        n_sims=5,
        gamma=gamma,
    )
    mcts = MCTS(**mcts_kwargs)

    df_sim = []
    for s in tqdm(range(0, n_columns * n_rows), desc="MCTS simulation"):
        d = distances_to_goal[s]
        sim_state_values = mcts.simulate_state_value(GridWorldNode(s, n_actions=4))

        for v in sim_state_values:
            df_sim.append(
                {"Distance to Goal": d, "State-Value": v, "Model": "Simulation"}
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x="Distance to Goal", y="State-Value", data=pd.DataFrame(df_sim), ax=ax
    )

    save_plot(fig, "mcts_simulation_variance.png")
    report.add_plot(
        fig,
        "MCTS Simulation Variance",
        "Variance in state-value estimates from MCTS simulations",
    )

    # Value Iteration Network
    print("Running Value Iteration Network...")
    gamma = 0.99
    state_action_reward_functions = e.get_state_action_reward_from_sucessor_rewards(
        reward_functions, transition_functions
    )

    kwargs = dict(
        transition_functions=transition_functions,
        reward_functions=state_action_reward_functions,
        n_rows=n_rows,
        n_columns=n_columns,
        gamma=gamma,
        iterations=1000,
        initialization_noise=0.01,
        return_interim_estimates=False,
    )

    _, V = ValueIterationNetwork.value_iteration(**kwargs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(V.reshape(20, 20))
    ax.set_title("Value Iteration Network Output")

    save_plot(fig, "value_iteration_network.png")
    report.add_plot(
        fig, "Value Iteration Network", "State values from Value Iteration Network"
    )

    # MCTS Policy Analysis
    report.add_section(
        "MCTS Policy Analysis",
        "Comparison of MCTS performance with different heuristic functions",
    )

    print("Running MCTS policy analysis...")
    start_state = (
        n_rows * n_columns - n_columns // 2 + n_columns // 4 - n_columns * (n_rows // 4)
    )
    start_node = GridWorldNode(start_state, n_actions=4)

    n_batches = 25
    beta = 5.0
    n_rollout = 500

    mcts_kwargs = dict(
        end_states=[goals_state],
        transition_functions=transition_functions,
        state_reward_function=reward_functions,
        exploration_weight=2**0.5 * 0.05,
        max_depth=500,
        n_sims=100,
        gamma=gamma,
    )

    batch_rollout_kwargs = dict(
        start_node=start_node,
        n_batches=n_batches,
        n_rollouts=n_rollout,
        beta=beta,
    )

    # MCTS with SR heuristic
    print("Running MCTS with SR heuristic...")
    mcts_sr = MctsSr(**mcts_kwargs)
    mcts_sr_selection_pmf, mcts_sr_visitations = mcts_sr.batch_rollouts(
        **batch_rollout_kwargs
    )

    # MCTS with Value function heuristic
    print("Running MCTS with Value function heuristic...")
    mcts_vi = MctsSr(**mcts_kwargs)
    mcts_vi.update_heuristic_function(V)
    mcts_vi_selection_pmf, mcts_vi_visitations = mcts_vi.batch_rollouts(
        **batch_rollout_kwargs
    )

    # Policy comparison
    pi_star = e.make_thread_the_needle_optimal_policy(n_rows, n_columns)

    def scoring_function(p, q):
        return p.dot(q)

    score_sr = [scoring_function(p, q) for p, q in zip(pi_star, mcts_sr_selection_pmf)]
    score_vi = [scoring_function(p, q) for p, q in zip(pi_star, mcts_vi_selection_pmf)]

    # Plot policy errors
    fig, ax = plt.subplots(figsize=(8, 6))
    neg = ax.imshow(np.reshape(score_sr, (n_rows, n_columns)), cmap="viridis")
    e.clean_up_thread_the_needle_plot(ax, n_columns, n_rows)
    plt.colorbar(neg, label="Policy Error")
    ax.set_title("Policy Error: MCTS w/ default heuristic")

    save_plot(fig, "mcts_policy_error.png")
    report.add_plot(
        fig,
        "MCTS Policy Error (Default)",
        "Policy error using MCTS with default heuristic",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    neg = ax.imshow(np.reshape(score_vi, (n_rows, n_columns)), cmap="viridis")
    e.clean_up_thread_the_needle_plot(ax, n_columns, n_rows)
    plt.colorbar(neg, label="Policy Error")
    ax.set_title("Policy Error: MCTS w/ True Value Function")

    save_plot(fig, "mcts_policy_error_true_value.png")
    report.add_plot(
        fig,
        "MCTS Policy Error (True Value)",
        "Policy error using MCTS with true value function heuristic",
    )

    # Expansion analysis
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mcts_vi_visitations.reshape(20, 20), cmap="magma_r")
    e.clean_up_thread_the_needle_plot(ax, n_rows, n_columns)
    plt.colorbar(im, label="Probability of Expansion")
    ax.set_title("Expanded States: MCTS w/ True Value Function")

    save_plot(fig, "mcts_expanded_states_true_value.png")
    report.add_plot(
        fig,
        "MCTS Expanded States (True Value)",
        "Probability of state expansion with true value function",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mcts_sr_visitations.reshape(20, 20), cmap="magma_r")
    e.clean_up_thread_the_needle_plot(ax, n_rows, n_columns)
    plt.colorbar(im, label="Probability of Expansion")
    ax.set_title("Expanded States: MCTS w/ default heuristic")

    save_plot(fig, "mcts_expanded_states.png")
    report.add_plot(
        fig,
        "MCTS Expanded States (Default)",
        "Probability of state expansion with default heuristic",
    )

    report.close_section()

    # Save HTML report
    report.save("mcts_analysis_report.html")

    print("Analysis complete!")
    print("Plots saved to: images/")
    print("HTML report saved to: mcts_analysis_report.html")


if __name__ == "__main__":
    main()

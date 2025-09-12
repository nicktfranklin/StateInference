import numpy as np

from value_iteration.models.agents.base_agent import BaseAgent
from value_iteration.task.gridworld import GridWorldEnv


def score_model(model: BaseAgent):
    vec_env = model.get_env()
    env: GridWorldEnv = vec_env.envs[0]

    pi, _ = env.unwrapped.get_optimal_policy()
    n_states = env.get_wrapper_attr("n_states")
    map_height = env.get_wrapper_attr("observation_model").map_height

    pmf = model.get_policy_prob(
        vec_env,
        n_states=n_states,
        map_height=map_height,
        cnn=True,
    )

    room_1_mask = (np.arange(400) < 200) * (np.arange(400) % 20 < 10)
    room_2_mask = (np.arange(400) >= 200) * (np.arange(400) % 20 < 10)
    room_3_mask = np.arange(400) % 20 >= 10

    return {
        "policy_pmf": pmf,
        "score": np.sum(pi * pmf, axis=1).mean(),
        "score_room1": np.sum(pi[room_1_mask] * pmf[room_1_mask], axis=1).mean(),
        "score_room2": np.sum(pi[room_2_mask] * pmf[room_2_mask], axis=1).mean(),
        "score_room3": np.sum(pi[room_3_mask] * pmf[room_3_mask], axis=1).mean(),
    }

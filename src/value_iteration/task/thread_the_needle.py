from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse


def define_valid_lattice_transitions(
    n_rows: int, n_columns: int, self_transitions: bool = False
) -> np.ndarray:
    """
    Defines a random transition matrix over a square lattice grid where transitions
    are valid only between neighboring states
    """

    n_states = n_rows * n_columns

    up_transitions = np.concatenate(
        [np.zeros(n_columns), np.ones(n_columns * (n_rows - 1))]
    ).reshape(-1, 1)

    left_transitions = np.concatenate(
        [np.zeros(1), np.ones(n_columns - 1)] * n_rows
    ).reshape(-1, 1)

    self_transitions = np.ones((n_states, 1)) * int(self_transitions)

    right_transitions = np.concatenate(
        [np.ones(n_columns - 1), np.zeros(1)] * n_rows
    ).reshape(-1, 1)

    down_transitions = np.concatenate(
        [np.ones(n_columns * (n_rows - 1)), np.zeros(n_columns)]
    ).reshape(-1, 1)

    return np.concatenate(
        [
            up_transitions,
            left_transitions,
            self_transitions,
            right_transitions,
            down_transitions,
        ],
        axis=1,
    )


def make_diagonal_matrix(
    diagonals: np.ndarray, k: int, sparse: bool = False
) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    if sparse:
        return scipy.sparse.diags(diagonals, offsets=k)
    return np.diag(diagonals, k=k)


def convert_movements_to_transition_matrix(
    movements: np.ndarray, n_cols: int, sparse=False
) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    transition_matrix = make_diagonal_matrix(movements[:, 2], k=0, sparse=sparse)
    transition_matrix += make_diagonal_matrix(movements[:-1, 3], k=1, sparse=sparse)
    transition_matrix += make_diagonal_matrix(movements[1:, 1], k=-1, sparse=sparse)

    transition_matrix += make_diagonal_matrix(
        movements[n_cols:, 0], k=-n_cols, sparse=sparse
    )
    transition_matrix += make_diagonal_matrix(
        movements[:-n_cols, 4], k=n_cols, sparse=sparse
    )
    return transition_matrix


def draw_dirichlet(valid_movements: np.ndarray) -> np.ndarray:
    _p = np.random.dirichlet(np.ones(np.sum(valid_movements)))
    p = np.zeros_like(valid_movements, dtype=float)
    t = 0
    for ii, isValid in enumerate(valid_movements):
        if isValid:
            p[ii] = _p[t]
            t += 1
    return p


def draw_dirichlet_transitions(movements: np.ndarray) -> np.ndarray:
    return np.array(list(map(draw_dirichlet, np.array(movements, dtype=bool))))


def make_diffusion_transition(movements: np.ndarray) -> np.ndarray:
    def _normalize(x):
        return x / np.sum(x)

    return np.array(list(map(_normalize, movements)))


def sample_random_transition_matrix(
    n_rows: int, n_columns: int, sparse=False
) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    movements = define_valid_lattice_transitions(n_rows, n_columns)
    movement_probs = draw_dirichlet_transitions(movements)
    return convert_movements_to_transition_matrix(
        movement_probs, n_columns, sparse=sparse
    )


def make_diffision_transition_matrix(
    n_rows: int, n_columns: int, sparse=False
) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    movements = define_valid_lattice_transitions(n_rows, n_columns)
    movement_probs = make_diffusion_transition(movements)
    return convert_movements_to_transition_matrix(
        movement_probs, n_columns, sparse=sparse
    )


def make_cardinal_movements_prob(
    n_rows: int,
    n_columns: int,
    slip_probability: float = 0.05,
    random_movement_on_error: bool = True,
) -> List[np.ndarray]:
    movements = define_valid_lattice_transitions(n_rows, n_columns)

    # if random_movement_on_error:
    diffused_movements = make_diffusion_transition(movements)

    def _map_movement(x, index):
        if np.isclose(x[index], 0):
            if random_movement_on_error:
                return x
            return np.array([0, 0, 1, 0, 0])

        output = np.array(x) * slip_probability
        output[index] += 1 - slip_probability
        return list(output)

    return [
        np.array(list(map(lambda x: _map_movement(x, ii), diffused_movements)))
        for ii in [0, 1, 3, 4]
    ]


def make_cardinal_transition_matrix(
    n_rows: int,
    n_columns: int,
    slip_probability: float = 0.05,
    sparse: bool = True,
    random_movement_on_error: bool = True,
) -> List[Union[np.ndarray, scipy.sparse.csr_matrix]]:
    cardinal_movements = make_cardinal_movements_prob(
        n_rows, n_columns, slip_probability, random_movement_on_error
    )
    return [
        convert_movements_to_transition_matrix(move_probs, n_columns, sparse=sparse)
        for move_probs in cardinal_movements
    ]


def renormalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.sum(vector)


def add_wall_between_two_states(
    state_a: int,
    state_b: int,
    transition_functions: List[Union[np.ndarray, scipy.sparse.csr_matrix]],
) -> List[Union[np.ndarray, scipy.sparse.csr_matrix]]:
    is_sparse = isinstance(transition_functions[0], scipy.sparse.csr_matrix)

    n_actions = len(transition_functions)
    modified_transitions = []
    for a0 in range(n_actions):
        t = transition_functions[a0]
        if is_sparse:
            t = t.toarray()
        t[state_a, state_b] = 0
        t[state_b, state_a] = 0
        t[state_a, :] = renormalize(t[state_a, :])
        t[state_b, :] = renormalize(t[state_b, :])

        if is_sparse:
            t = scipy.sparse.csr_matrix(t)

        modified_transitions.append(t)
    return modified_transitions


def _check_valid(pos, max_pos):
    return (pos > -1) and (pos < max_pos)


def get_state_action_reward_from_sucessor_rewards(
    reward_function_over_sucessors: np.ndarray,
    transitions: List[Union[np.ndarray, scipy.sparse.csr_matrix]],
) -> List[Union[np.ndarray, scipy.sparse.csr_matrix]]:
    reward_function_over_sa = [
        t.dot(reward_function_over_sucessors) for t in transitions
    ]
    return reward_function_over_sa


class GridWorld:
    def __init__(
        self,
        n_rows: int,
        n_columns: int,
        transition_functions: List[Union[np.ndarray, scipy.sparse.csr_matrix]],
        state_reward_function: np.ndarray,
    ):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.transition_functions = transition_functions
        self.state_reward_function = state_reward_function
        self.state_action_reward_functions = (
            get_state_action_reward_from_sucessor_rewards(
                state_reward_function, transition_functions
            )
        )

    @staticmethod
    def get_state_from_position(row: int, column: int, n_columns: int) -> int:
        return n_columns * row + column

    @staticmethod
    def get_position_from_state(state: int, n_columns: int) -> Tuple[int, int]:
        row = state // n_columns
        column = state % n_columns
        return row, column

    @staticmethod
    def get_neighbors(state: int, n_columns: int, n_rows: int) -> List[int]:
        r, c = GridWorld.get_position_from_state(state, n_columns)
        neighbors = []
        for dr, dc in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
            if _check_valid(r + dr, n_rows) and _check_valid(c + dc, n_columns):
                neighbors.append(
                    GridWorld.get_state_from_position(r + dr, c + dc, n_columns)
                )
        return neighbors


def make_thread_the_needle_walls(n_columns: int) -> List[List[int]]:
    list_walls = []
    for ii in range(0, n_columns // 2 - 1):
        wall = [
            n_columns * (n_columns // 2 - 1) + ii,
            n_columns * (n_columns // 2) + ii,
        ]
        list_walls.append(wall)

    for ii in range(0, n_columns - 1):
        list_walls.append(
            [n_columns * ii + (n_columns // 2) - 1, n_columns * ii + n_columns // 2]
        )
    return list_walls


def make_thread_the_needle_walls_moved_door(n_columns: int) -> List[List[int]]:
    list_walls = []
    for ii in range(0, n_columns // 2 - 0):
        wall = [
            n_columns * (n_columns // 2 - 1) + ii,
            n_columns * (n_columns // 2) + ii,
        ]
        list_walls.append(wall)

    for ii in range(1, n_columns - 1):
        list_walls.append(
            [n_columns * ii + (n_columns // 2) - 1, n_columns * ii + n_columns // 2]
        )
    return list_walls


def make_thread_the_needle_optimal_policy(n_rows: int, n_columns: int) -> np.ndarray:
    _N_ACTIONS = 4
    optimal_policy = np.zeros((n_rows * n_columns, _N_ACTIONS), dtype=int)

    # In the first room, along the left wall
    c = n_columns // 2
    for r in range(n_rows - 1):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([0, 0, 0, 1])

    # in the first room, along the bottom
    r = n_rows - 1
    for c in range(n_columns // 2, n_columns):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([0, 1, 0, 0])

    # everywhere else in the first room
    for r in range(n_rows - 1):
        for c in range(n_columns // 2 + 1, n_columns):
            state = GridWorld.get_state_from_position(r, c, n_columns)
            optimal_policy[state, :] = np.array([0, 1, 0, 1])

    # first spot in second room
    r = n_rows - 1
    c = n_columns // 2 - 1
    state = GridWorld.get_state_from_position(r, c, n_columns)
    optimal_policy[state, :] = np.array([1, 0, 0, 0])

    # along right wall in second room
    c = n_columns // 2 - 1
    for r in range(n_rows // 2, n_rows - 1):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([1, 0, 0, 0])

    # along top wall in second room
    r = n_rows // 2
    for c in range(n_columns // 2 - 1):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([0, 0, 1, 0])

    # everywhere else in the second room
    for r in range(n_rows // 2 + 1, n_rows):
        for c in range(n_columns // 2 - 1):
            state = GridWorld.get_state_from_position(r, c, n_columns)
            optimal_policy[state, :] = np.array([1, 0, 1, 0])

    # everywhere else
    for r in range(n_rows // 2):
        for c in range(n_columns // 2):
            state = GridWorld.get_state_from_position(r, c, n_columns)
            optimal_policy[state, :] = np.array([1, 1, 0, 0])

    # goal state (overwrites previous value)
    optimal_policy[0, :] = np.ones(4)

    return optimal_policy


def make_thread_the_needle_with_doors_optimal_policy(
    n_rows: int, n_columns: int
) -> np.ndarray:
    _N_ACTIONS = 4
    optimal_policy = np.zeros((n_rows * n_columns, _N_ACTIONS), dtype=int)
    # i think the action key is [up, left, right, down]

    # In the first room, along the left wall
    c = n_columns // 2
    for r in range(n_rows - 1):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([1, 1, 0, 0])

    # in the first room, along the bottom
    r = n_rows - 1
    for c in range(n_columns // 2, n_columns):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([1, 1, 0, 0])

    # everywhere else in the first room
    for r in range(n_rows - 1):
        for c in range(n_columns // 2 + 1, n_columns):
            state = GridWorld.get_state_from_position(r, c, n_columns)
            optimal_policy[state, :] = np.array([1, 1, 0, 0])

    # first spot in second room
    r = n_rows - 1
    c = n_columns // 2 - 1
    state = GridWorld.get_state_from_position(r, c, n_columns)
    optimal_policy[state, :] = np.array([0, 0, 1, 0])

    # along right wall in second room
    c = n_columns // 2 - 1
    for r in range(n_rows // 2, n_rows - 1):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([0, 0, 0, 1])

    # along top wall in second room
    r = n_rows // 2
    for c in range(n_columns // 2 - 1):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([0, 0, 1, 1])

    # everywhere else in the second room
    for r in range(n_rows // 2 + 1, n_rows):
        for c in range(n_columns // 2 - 1):
            state = GridWorld.get_state_from_position(r, c, n_columns)
            optimal_policy[state, :] = np.array([0, 0, 1, 1])

    # everywhere else
    for r in range(n_rows // 2):
        for c in range(n_columns // 2):
            state = GridWorld.get_state_from_position(r, c, n_columns)
            optimal_policy[state, :] = np.array([1, 1, 0, 0])

    # top row
    c = 0
    for r in range(n_rows):
        state = GridWorld.get_state_from_position(r, c, n_columns)
        optimal_policy[state, :] = np.array([0, 1, 0, 0])

    # goal state (overwrites previous value)
    optimal_policy[0, :] = np.ones(4)

    # special case -- top-right and bottom left corners the optimal policy is random
    r, c = 0, n_columns - 1
    state = GridWorld.get_state_from_position(r, c, n_columns)
    optimal_policy[state, :] = 1

    r, c = n_rows - 1, 0
    state = GridWorld.get_state_from_position(r, c, n_columns)
    optimal_policy[state, :] = 1

    return optimal_policy


def make_thread_the_needle(
    n_rows: int,
    n_columns: int,
    slip_probability: float = 0.05,
    movement_penalty: float = -0.01,
    sparse: bool = True,
    random_movement_on_error: bool = True,
    list_walls: Optional[List[List[int]]] = None,
) -> Tuple[List[Union[np.ndarray, scipy.sparse.csr_matrix]], np.ndarray, np.ndarray]:
    assert n_rows == n_columns, "Columns and Rows don't match!"
    assert n_columns >= 4, "Minimum size: 4x4"

    _N_ACTIONS = 4

    transition_functions = make_cardinal_transition_matrix(
        n_columns=n_columns,
        n_rows=n_rows,
        slip_probability=slip_probability,
        sparse=sparse,
        random_movement_on_error=random_movement_on_error,
    )

    if not list_walls:
        list_walls = make_thread_the_needle_walls(n_columns)

    for s0, s1 in list_walls:
        transition_functions = add_wall_between_two_states(s0, s1, transition_functions)

    # define the reward purely in terms of sucessor states
    state_reward_function = np.ones(n_rows * n_columns) * movement_penalty
    goals_state = 0
    state_reward_function[goals_state] = 1.0

    # define the optimal policy for the task
    optimal_policy = make_thread_the_needle_optimal_policy(n_rows, n_columns)

    return transition_functions, state_reward_function, optimal_policy


def make_thread_the_needle_diffusion_transitions(
    n_rows: int, n_columns: int, sparse: bool = True
):
    movements = define_valid_lattice_transitions(n_rows, n_columns)

    # if random_movement_on_error:
    diffused_movements = make_diffusion_transition(movements)

    return convert_movements_to_transition_matrix(diffused_movements, n_columns, sparse)


def clean_up_thread_the_needle_plot(
    ax, n_columns=8, n_rows=8, walls=None, wall_color="k"
):
    ax.set_xticks([])
    ax.set_yticks([])

    # plot the gridworld tiles
    for r in range(n_rows):
        ax.plot([-0.5, n_columns - 0.5], [r - 0.5, r - 0.5], c="grey", lw=0.5)
    for c in range(n_columns):
        ax.plot([c - 0.5, c - 0.5], [-0.5, n_rows - 0.5], c="grey", lw=0.5)

    if not walls:
        walls = make_thread_the_needle_walls(n_columns)

    for s0, s1 in walls:
        r0, c0 = GridWorld.get_position_from_state(s0, n_columns)
        r1, c1 = GridWorld.get_position_from_state(s1, n_columns)

        x = (r0 + r1) / 2
        y = (c0 + c1) / 2

        assert (r0 == r1) or (c0 == c1), f"Not a valid wall! {r0} {r1} {c0} {s1}"
        if c0 == c1:
            ax.plot([y - 0.5, y + 0.5], [x, x], c=wall_color, lw=3)
        else:
            ax.plot([y, y], [x - 0.5, x + 0.5], c=wall_color, lw=3)


def one_d_reward_at_one_end(
    n_columns: int,
    slip_probability: float = 0.05,
    movement_penalty: float = -0.01,
    sparse: bool = True,
    random_movement_on_error: bool = False,
) -> Tuple[List[Union[np.ndarray, scipy.sparse.csr_matrix]], np.ndarray, np.ndarray]:
    transition_functions = make_cardinal_transition_matrix(
        n_columns=n_columns,
        n_rows=1,
        slip_probability=slip_probability,
        sparse=sparse,
        random_movement_on_error=random_movement_on_error,
    )

    # we only want the left / right actions
    transition_functions = [transition_functions[1], transition_functions[2]]

    # define the reward purely in terms of sucessor states
    state_reward_function = np.ones(n_columns) * movement_penalty
    goals_state = 0
    state_reward_function[goals_state] = 1.0
    state_reward_function[n_columns - 1] = -1.0

    # only two actions
    optimal_policy = np.zeros((n_columns, 2), dtype=int)
    optimal_policy[:, 0] = 1
    optimal_policy[goals_state, :] = 1
    #
    return transition_functions, state_reward_function, optimal_policy


def clean_up_reward_at_end(ax, n_columns=51):
    ax.set_xticks([])
    ax.set_yticks([])
    for c in range(n_columns):
        ax.plot([c - 0.5, c - 0.5], [1 - 0.5, 0 - 0.5], c="grey", lw=0.5)


def find_sortest_path_length(
    state_value_function: np.ndarray,
    transition_functions: List[np.ndarray],
    goal_state: int,
    start_state: int,
    n_columns: int,
    n_rows: int,
) -> int:
    path = find_shortest_path(
        state_value_function,
        transition_functions,
        goal_state,
        start_state,
        n_columns,
        n_rows,
    )
    return len(path)


def find_shortest_path(
    state_value_function: np.ndarray,
    transition_functions: List[np.ndarray],
    goal_state: int,
    start_state: int,
    n_columns: int,
    n_rows: int,
) -> np.ndarray:
    state = start_state
    path = []

    t_avg = np.mean(transition_functions, axis=0)

    def filter_values(start_state, end_state, end_state_value):
        if t_avg[start_state][end_state] > 0:
            return end_state_value
        return 0

    while state is not goal_state:
        neighbors = GridWorld.get_neighbors(state, n_columns, n_rows)
        values = {
            n: filter_values(state, n, state_value_function[n])
            for n in neighbors
            if n not in path
        }
        state = max(values, key=values.get)
        path.append(state)

    return path

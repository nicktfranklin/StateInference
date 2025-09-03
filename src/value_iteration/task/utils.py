from typing import Tuple, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
StateType = TypeVar("StateType")
RewType = TypeVar("RewType")


def get_state_from_position(row: int, column: int, n_columns: int) -> int:
    return n_columns * row + column


def get_position_from_state(state: int, n_columns: int) -> Tuple[int, int]:
    row = state // n_columns
    column = state % n_columns
    return row, column

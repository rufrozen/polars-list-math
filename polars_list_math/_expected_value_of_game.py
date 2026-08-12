from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
from polars.plugins import register_plugin_function

from ._list_similarity import _native_library_available, _parse_into_expr, _validate_polars_input

PolarsGameInput = pl.Expr | str
_LIB = Path(__file__).parent


def expected_value_of_game(game: PolarsGameInput) -> pl.Expr:
    """Compute the expected value of a sequential game for each Polars row.

    A game is a list of states, each state is a list of actions, and each action
    has ``probability``, ``value``, and ``next_state`` fields.
    """
    _validate_polars_input(game, "game")
    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="expected_value_of_game",
            args=[game],
            is_elementwise=True,
            use_abs_path=True,
        )
    return _parse_into_expr(game).map_elements(
        _expected_value_of_game_row,
        return_dtype=pl.Float64,
        skip_nulls=False,
    )


def _expected_value_of_game_row(game: Any) -> float:
    states = _as_list(game, "game")
    n = len(states)
    if n < 2:
        return 0.0
    if _as_list(states[n - 1], "final state"):
        msg = "the final state must have no actions"
        raise ValueError(msg)

    evog = [0.0] * n
    for i in range(n - 2, -1, -1):
        evos = 0.0
        actions = _as_list(states[i], f"state {i}")
        for action in actions:
            probability, value, next_state = _action_values(action)
            if next_state <= i or next_state >= n:
                msg = f"next_state in state {i} must be greater than {i} and less than {n}"
                raise ValueError(msg)
            evos += probability * (value + evog[next_state])
        evog[i] = evos
    return evog[0]


def _as_list(value: Any, name: str) -> list[Any]:
    if isinstance(value, pl.Series):
        return value.to_list()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        msg = f"{name} must be a list"
        raise TypeError(msg)
    return list(value)


def _action_values(action: Any) -> tuple[float, float, int]:
    if isinstance(action, Mapping):
        values = (action["probability"], action["value"], action["next_state"])
    else:
        values = tuple(action)
    probability, value, next_state = values
    return float(probability), float(value), int(next_state)

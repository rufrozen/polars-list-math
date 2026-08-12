from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from polars_list_math import _expected_value_of_game, expected_value_of_game

ACTION_DTYPE = pl.Struct({"probability": pl.Float64, "value": pl.Float64, "next_state": pl.Int64})
GAME_DTYPE = pl.List(pl.List(ACTION_DTYPE))


def _reference(game: list[list[tuple[float, float, int]]]) -> float:
    n = len(game)
    if n < 2:
        return 0.0
    assert game[n - 1] == []
    evog = [0.0] * n
    for i in range(n - 2, -1, -1):
        evos = 0.0
        for prob, value, j in game[i]:
            assert j > i
            evos += prob * (value + evog[j])
        evog[i] = evos
    return evog[0]


def _records(game: list[list[tuple[float, float, int]]]) -> list[list[dict[str, Any]]]:
    return [
        [
            {"probability": probability, "value": value, "next_state": next_state}
            for probability, value, next_state in state
        ]
        for state in game
    ]


def _native_available() -> bool:
    try:
        pl.DataFrame({"game": [[]]}, schema={"game": GAME_DTYPE}).select(
            expected_value_of_game("game")
        )
    except BaseException:
        return False
    return True


GAMES = [
    [],
    [[]],
    [[(1.0, 5.0, 1)], []],
    [[(0.5, 2.0, 1), (0.5, 10.0, 2)], [(1.0, 3.0, 2)], []],
    [[(0.25, -4.0, 2), (0.75, 2.0, 1)], [(1.0, 1.0, 2)], []],
]


def test_python_fallback_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_expected_value_of_game, "_native_library_available", lambda: False)
    frame = pl.DataFrame({"game": [_records(game) for game in GAMES]}, schema={"game": GAME_DTYPE})
    result = frame.select(expected_value_of_game("game"))["game"].to_list()
    assert result == pytest.approx([_reference(game) for game in GAMES])


def test_invalid_game_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_expected_value_of_game, "_native_library_available", lambda: False)
    game = [[(1.0, 1.0, 0)], []]
    frame = pl.DataFrame({"game": [_records(game)]}, schema={"game": GAME_DTYPE})
    with pytest.raises(Exception, match="next_state in state 0"):
        frame.select(expected_value_of_game("game"))


@pytest.mark.skipif(not _native_available(), reason="native plugin has not been built")
def test_native_matches_reference() -> None:
    frame = pl.DataFrame({"game": [_records(game) for game in GAMES]}, schema={"game": GAME_DTYPE})
    result = frame.select(expected_value_of_game("game"))["game"].to_list()
    assert result == pytest.approx([_reference(game) for game in GAMES])

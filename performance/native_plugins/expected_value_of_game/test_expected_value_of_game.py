import polars as pl
import polars_list_math as plm
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)


def test_expected_value_of_game_performance() -> None:
    game = [
        [
            {"probability": 0.6, "value": 2.0, "next_state": 1},
            {"probability": 0.4, "value": 1.0, "next_state": 2},
        ],
        [{"probability": 1.0, "value": 3.0, "next_state": 2}],
        [],
    ]
    frame = pl.DataFrame({"game": [game for _ in range(ROW_COUNT)]})
    run_case(
        PerformanceCase(
            "expected_value_of_game",
            plugin_module("_expected_value_of_game"),
            frame,
            lambda: plm.expected_value_of_game("game"),
        )
    )


def test_expected_value_of_game_complex_performance() -> None:
    state_count = 16
    game = [
        [
            {
                "probability": probability,
                "value": float(state_index + offset),
                "next_state": min(state_index + offset, state_count - 1),
            }
            for offset, probability in ((1, 0.5), (2, 0.3), (3, 0.2))
            if state_index + offset < state_count
        ]
        for state_index in range(state_count - 1)
    ]
    game.append([])
    frame = pl.DataFrame({"game": [game for _ in range(ROW_COUNT)]})
    run_case(
        PerformanceCase(
            "expected_value_of_game_complex",
            plugin_module("_expected_value_of_game"),
            frame,
            lambda: plm.expected_value_of_game("game"),
        )
    )

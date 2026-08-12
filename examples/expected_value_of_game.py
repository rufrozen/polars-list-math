"""Compute the expected value of a sequential game."""

import polars as pl
from polars_list_math import expected_value_of_game


def main() -> None:
    frame = pl.DataFrame(
        {
            "game": [
                [
                    [
                        {"probability": 0.5, "value": 2.0, "next_state": 1},
                        {"probability": 0.5, "value": 10.0, "next_state": 2},
                    ],
                    [{"probability": 1.0, "value": 3.0, "next_state": 2}],
                    [],
                ]
            ]
        }
    )
    print(frame.select(expected_value_of_game("game")))


if __name__ == "__main__":
    main()

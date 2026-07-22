from __future__ import annotations

import polars as pl
import polars_list_math  # noqa: F401
import pytest
from polars_list_math import _list_combinations


def _native_list_combinations_available() -> bool:
    if not _list_combinations._native_library_available():
        return False

    df = pl.DataFrame({"a": [[1, 2]], "b": [[3]]})
    try:
        df.select(pl.col("a").list.combinations().alias("pairs"))
        df.select(pl.col("a").list.combinations_to("b").alias("pairs"))
    except Exception:
        return False
    return True


def test_list_combinations_returns_list_of_structs() -> None:
    df = pl.DataFrame(
        {"a": [[10, 20, 30], [1], [], None, [None, 2]]},
        schema={"a": pl.List(pl.Int64)},
    )

    out = df.select(pl.col("a").list.combinations().alias("pairs"))

    assert out["pairs"].to_list() == [
        [
            {"left_value": 10, "right_value": 10},
            {"left_value": 10, "right_value": 20},
            {"left_value": 10, "right_value": 30},
            {"left_value": 20, "right_value": 20},
            {"left_value": 20, "right_value": 30},
            {"left_value": 30, "right_value": 30},
        ],
        [{"left_value": 1, "right_value": 1}],
        [],
        None,
        [
            {"left_value": None, "right_value": None},
            {"left_value": None, "right_value": 2},
            {"left_value": 2, "right_value": 2},
        ],
    ]


def test_list_combinations_supports_with_index() -> None:
    df = pl.DataFrame({"a": [[10, 20]]}, schema={"a": pl.List(pl.Int64)})

    out = df.select(pl.col("a").list.combinations(with_index=True).alias("pairs"))

    assert out["pairs"].to_list() == [
        [
            {"left_index": 0, "right_index": 0, "left_value": 10, "right_value": 10},
            {"left_index": 0, "right_index": 1, "left_value": 10, "right_value": 20},
            {"left_index": 1, "right_index": 1, "left_value": 20, "right_value": 20},
        ]
    ]


def test_list_combinations_includes_indexes_when_index_name_is_given() -> None:
    df = pl.DataFrame({"a": [[10]]}, schema={"a": pl.List(pl.Int64)})

    out = df.select(pl.col("a").list.combinations(left_index="i").alias("pairs"))

    assert out["pairs"].to_list() == [
        [{"i": 0, "right_index": 0, "left_value": 10, "right_value": 10}]
    ]


def test_list_combinations_supports_custom_fields_and_skip_null() -> None:
    df = pl.DataFrame({"a": [[None, 2, 3]]}, schema={"a": pl.List(pl.Int64)})

    out = df.select(
        pl.col("a")
        .list.combinations(
            left_index="i",
            right_index="j",
            left_value="x",
            right_value="y",
            skip_null=True,
        )
        .alias("pairs")
    )

    assert out["pairs"].to_list() == [
        [
            {"i": 1, "j": 1, "x": 2, "y": 2},
            {"i": 1, "j": 2, "x": 2, "y": 3},
            {"i": 2, "j": 2, "x": 3, "y": 3},
        ]
    ]


def test_list_combinations_supports_shuffled_nested_list_values() -> None:
    df = pl.DataFrame(
        {
            "v": [
                [[3, 1], [2], [5, 4]],
            ]
        },
        schema={"v": pl.List(pl.List(pl.Int64))},
    )

    out = df.select(pl.col("v").list.combinations().alias("pairs"))

    assert out["pairs"].to_list() == [
        [
            {"left_value": [3, 1], "right_value": [3, 1]},
            {"left_value": [3, 1], "right_value": [2]},
            {"left_value": [3, 1], "right_value": [5, 4]},
            {"left_value": [2], "right_value": [2]},
            {"left_value": [2], "right_value": [5, 4]},
            {"left_value": [5, 4], "right_value": [5, 4]},
        ]
    ]


def test_list_combinations_skip_null_omits_null_pairs_and_keeps_indexes() -> None:
    df = pl.DataFrame(
        {"v": [[0, None, 1, 2, None], [None, None]]},
        schema={"v": pl.List(pl.Int64)},
    )

    out = df.select(pl.col("v").list.combinations(with_index=True, skip_null=True).alias("pairs"))

    assert out["pairs"].to_list() == [
        [
            {"left_index": 0, "right_index": 0, "left_value": 0, "right_value": 0},
            {"left_index": 0, "right_index": 2, "left_value": 0, "right_value": 1},
            {"left_index": 0, "right_index": 3, "left_value": 0, "right_value": 2},
            {"left_index": 2, "right_index": 2, "left_value": 1, "right_value": 1},
            {"left_index": 2, "right_index": 3, "left_value": 1, "right_value": 2},
            {"left_index": 3, "right_index": 3, "left_value": 2, "right_value": 2},
        ],
        [],
    ]


def test_list_combinations_to_returns_cross_pairs() -> None:
    df = pl.DataFrame(
        {
            "left": [[1, 2], [], None],
            "right": [["a", "b"], ["c"], ["d"]],
        },
        schema={"left": pl.List(pl.Int64), "right": pl.List(pl.String)},
    )

    out = df.select(pl.col("left").list.combinations_to("right").alias("pairs"))

    assert out["pairs"].to_list() == [
        [
            {"left_value": 1, "right_value": "a"},
            {"left_value": 1, "right_value": "b"},
            {"left_value": 2, "right_value": "a"},
            {"left_value": 2, "right_value": "b"},
        ],
        [],
        None,
    ]


def test_list_combinations_to_supports_custom_fields_and_skip_null() -> None:
    df = pl.DataFrame(
        {"left": [[None, 2]], "right": [[10, None]]},
        schema={"left": pl.List(pl.Int64), "right": pl.List(pl.Int64)},
    )

    out = df.select(
        pl.col("left")
        .list.combinations_to(
            "right",
            left_index="source_index",
            right_index="target_index",
            left_value="source",
            right_value="target",
            skip_null=True,
        )
        .alias("pairs")
    )

    assert out["pairs"].to_list() == [
        [{"source_index": 1, "target_index": 0, "source": 2, "target": 10}]
    ]


def test_list_combinations_to_supports_shuffled_nested_list_values() -> None:
    df = pl.DataFrame(
        {
            "left": [
                [[3, 1], [2]],
            ],
            "right": [
                [[8], [5, 4]],
            ],
        },
        schema={
            "left": pl.List(pl.List(pl.Int64)),
            "right": pl.List(pl.List(pl.Int64)),
        },
    )

    out = df.select(pl.col("left").list.combinations_to("right").alias("pairs"))

    assert out["pairs"].to_list() == [
        [
            {"left_value": [3, 1], "right_value": [8]},
            {"left_value": [3, 1], "right_value": [5, 4]},
            {"left_value": [2], "right_value": [8]},
            {"left_value": [2], "right_value": [5, 4]},
        ]
    ]


def test_list_combinations_to_skip_null_omits_null_pairs_and_keeps_indexes() -> None:
    df = pl.DataFrame(
        {
            "left": [[0, None, 1, 2, None], [None, None]],
            "right": [[10, None, 11], [None]],
        },
        schema={"left": pl.List(pl.Int64), "right": pl.List(pl.Int64)},
    )

    out = df.select(
        pl.col("left").list.combinations_to("right", with_index=True, skip_null=True).alias("pairs")
    )

    assert out["pairs"].to_list() == [
        [
            {"left_index": 0, "right_index": 0, "left_value": 0, "right_value": 10},
            {"left_index": 0, "right_index": 2, "left_value": 0, "right_value": 11},
            {"left_index": 2, "right_index": 0, "left_value": 1, "right_value": 10},
            {"left_index": 2, "right_index": 2, "left_value": 1, "right_value": 11},
            {"left_index": 3, "right_index": 0, "left_value": 2, "right_value": 10},
            {"left_index": 3, "right_index": 2, "left_value": 2, "right_value": 11},
        ],
        [],
    ]


def test_list_combinations_validation_errors() -> None:
    with pytest.raises(TypeError, match="field names"):
        pl.col("a").list.combinations(left_index=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unique"):
        pl.col("a").list.combinations(left_index="same", right_index="same")

    with pytest.raises(TypeError, match="skip_null"):
        pl.col("a").list.combinations(skip_null="yes")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="with_index"):
        pl.col("a").list.combinations(with_index="yes")  # type: ignore[arg-type]


@pytest.mark.skipif(
    not _native_list_combinations_available(),
    reason="native list combinations plugin has not been built",
)
def test_native_list_combinations_rejects_non_list_inputs() -> None:
    df = pl.DataFrame({"a": [1], "b": [[1]]})

    with pytest.raises(pl.exceptions.ComputeError, match="expected `List`"):
        df.select(pl.col("a").list.combinations())

    with pytest.raises(pl.exceptions.ComputeError, match="expected `List`"):
        df.select(pl.col("b").list.combinations_to("a"))

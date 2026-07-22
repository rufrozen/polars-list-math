from __future__ import annotations

import polars as pl
import polars_list_math  # noqa: F401
import pytest
from polars_list_math import _list_mean_similarity


def _swapped_score() -> float:
    return (0.9 + 0.9 + 0.81) / (1.0 + 1.0 + 0.81)


def _two_item_swapped_score() -> float:
    return (0.9 + 0.9) / (1.0 + 1.0)


def _native_list_mean_similarity_available() -> bool:
    if not _list_mean_similarity._native_library_available():
        return False

    df = pl.DataFrame(
        {"lists": [[[1], [1]]]},
        schema={"lists": pl.List(pl.List(pl.Int64))},
    )
    try:
        df.select(pl.col("lists").list.mean_similarity().alias("similarity"))
        df.select(pl.col("lists").list.mean_similarity_to("lists").alias("similarity"))
    except Exception:
        return False
    return True


def test_list_mean_similarity_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_list_mean_similarity, "_native_library_available", lambda: False)
    df = pl.DataFrame(
        {
            "lists": [
                [[1, 2, 3], [1, 2, 3], [4, 5, 6]],
                [[1, 2, 3]],
                [],
                None,
                [[1, 2, 3], None, [1, 2, 3]],
            ]
        },
        schema={"lists": pl.List(pl.List(pl.Int64))},
    )

    out = df.select(pl.col("lists").list.mean_similarity().alias("similarity"))
    values = out["similarity"].to_list()

    assert values[0] == pytest.approx([0.5, 0.5, 0.0])
    assert values[1] == [None]
    assert values[2] == []
    assert values[3] is None
    assert values[4] == pytest.approx([1.0, None, 1.0])


def test_list_mean_similarity_to_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_list_mean_similarity, "_native_library_available", lambda: False)
    df = pl.DataFrame(
        {
            "lists_1": [
                [[1, 2, 3], [4, 5, 6]],
                [],
                None,
                [[1, 2, 3], None],
                [[1, 2, 3]],
            ],
            "lists_2": [
                [[1, 2, 3], [2, 1, 3], [7, 8, 9]],
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3]],
                [[1, 2, 3], None, [4, 5, 6]],
                None,
            ],
        },
        schema={
            "lists_1": pl.List(pl.List(pl.Int64)),
            "lists_2": pl.List(pl.List(pl.Int64)),
        },
    )

    out = df.select(pl.col("lists_2").list.mean_similarity_to("lists_1").alias("similarity"))
    values = out["similarity"].to_list()

    assert values[0] == pytest.approx([0.5, _swapped_score() / 2.0, 0.0])
    assert values[1] == [None, None]
    assert values[2] == [None]
    assert values[3] == pytest.approx([1.0, None, 0.0])
    assert values[4] is None


def test_list_mean_similarity_python_fallback_accepts_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_list_mean_similarity, "_native_library_available", lambda: False)
    df = pl.DataFrame(
        {"lists": [[["a", "b", "c"], ["b", "a", "c"]]]},
        schema={"lists": pl.List(pl.List(pl.String))},
    )

    out = df.select(pl.col("lists").list.mean_similarity().alias("similarity"))

    assert out["similarity"].to_list()[0] == pytest.approx([_swapped_score(), _swapped_score()])


def test_list_mean_similarity_python_fallback_accepts_primitive_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_list_mean_similarity, "_native_library_available", lambda: False)
    df = pl.DataFrame(
        {"lists": [[[True, False], [False, True]]]},
        schema={"lists": pl.List(pl.List(pl.Boolean))},
    )

    out = df.select(pl.col("lists").list.mean_similarity().alias("similarity"))

    assert out["similarity"].to_list()[0] == pytest.approx(
        [_two_item_swapped_score(), _two_item_swapped_score()]
    )


@pytest.mark.skipif(
    not _native_list_mean_similarity_available(),
    reason="native list_mean_similarity plugin has not been built",
)
def test_native_list_mean_similarity_plugin_matches_fallback() -> None:
    df = pl.DataFrame(
        {
            "lists": [
                [[1, 2, 3], [1, 2, 3], [4, 5, 6]],
                [[1, 2, 3]],
                [],
                None,
                [[1, 2, 3], None, [1, 2, 3]],
            ]
        },
        schema={"lists": pl.List(pl.List(pl.Int64))},
    )

    out = df.select(pl.col("lists").list.mean_similarity().alias("similarity"))
    values = out["similarity"].to_list()

    assert values[0] == pytest.approx([0.5, 0.5, 0.0])
    assert values[1] == [None]
    assert values[2] == []
    assert values[3] is None
    assert values[4] == pytest.approx([1.0, None, 1.0])


@pytest.mark.skipif(
    not _native_list_mean_similarity_available(),
    reason="native list_mean_similarity plugin has not been built",
)
def test_native_list_mean_similarity_to_plugin_matches_fallback() -> None:
    df = pl.DataFrame(
        {
            "lists_1": [
                [[1, 2, 3], [4, 5, 6]],
                [],
                None,
                [[1, 2, 3], None],
                [[1, 2, 3]],
            ],
            "lists_2": [
                [[1, 2, 3], [2, 1, 3], [7, 8, 9]],
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3]],
                [[1, 2, 3], None, [4, 5, 6]],
                None,
            ],
        },
        schema={
            "lists_1": pl.List(pl.List(pl.Int64)),
            "lists_2": pl.List(pl.List(pl.Int64)),
        },
    )

    out = df.select(pl.col("lists_2").list.mean_similarity_to("lists_1").alias("similarity"))
    values = out["similarity"].to_list()

    assert values[0] == pytest.approx([0.5, _swapped_score() / 2.0, 0.0])
    assert values[1] == [None, None]
    assert values[2] == [None]
    assert values[3] == pytest.approx([1.0, None, 0.0])
    assert values[4] is None


@pytest.mark.skipif(
    not _native_list_mean_similarity_available(),
    reason="native list_mean_similarity plugin has not been built",
)
def test_native_list_mean_similarity_plugin_accepts_strings() -> None:
    df = pl.DataFrame(
        {
            "lists_1": [[["a", "b", "c"]]],
            "lists_2": [[["b", "a", "c"], ["x"]]],
        },
        schema={
            "lists_1": pl.List(pl.List(pl.String)),
            "lists_2": pl.List(pl.List(pl.String)),
        },
    )

    out = df.select(pl.col("lists_2").list.mean_similarity_to("lists_1").alias("similarity"))

    assert out["similarity"].to_list()[0] == pytest.approx([_swapped_score(), 0.0])


@pytest.mark.skipif(
    not _native_list_mean_similarity_available(),
    reason="native list_mean_similarity plugin has not been built",
)
def test_native_list_mean_similarity_plugin_accepts_primitive_lists() -> None:
    df = pl.DataFrame(
        {"lists": [[[True, False], [False, True], [True, False]]]},
        schema={"lists": pl.List(pl.List(pl.Boolean))},
    )

    out = df.select(pl.col("lists").list.mean_similarity().alias("similarity"))

    assert out["similarity"].to_list()[0] == pytest.approx([0.95, _two_item_swapped_score(), 0.95])


@pytest.mark.skipif(
    not _native_list_mean_similarity_available(),
    reason="native list_mean_similarity plugin has not been built",
)
def test_native_list_mean_similarity_to_plugin_accepts_primitive_lists() -> None:
    df = pl.DataFrame(
        {
            "lists_1": [[[1.5, 2.5, 3.5]]],
            "lists_2": [[[2.5, 1.5, 3.5], [9.5]]],
        },
        schema={
            "lists_1": pl.List(pl.List(pl.Float64)),
            "lists_2": pl.List(pl.List(pl.Float64)),
        },
    )

    out = df.select(pl.col("lists_2").list.mean_similarity_to("lists_1").alias("similarity"))

    assert out["similarity"].to_list()[0] == pytest.approx([_swapped_score(), 0.0])

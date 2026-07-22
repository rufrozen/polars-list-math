from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import polars as pl
import polars_list_math  # noqa: F401
import pytest
from polars_list_math import _list_similarity, py_list_similarity

DocumentValue = Any


def _score(
    list_a: Sequence[DocumentValue],
    list_b: Sequence[DocumentValue],
    p: float = 0.9,
) -> float:
    return py_list_similarity(list_a, list_b, p=p)


def _swapped_score() -> float:
    return (0.9 + 0.9 + 0.81) / (1.0 + 1.0 + 0.81)


def _two_item_swapped_score() -> float:
    return (0.9 + 0.9) / (1.0 + 1.0)


def _native_list_similarity_available() -> bool:
    if not _list_similarity._native_library_available():
        return False

    df = pl.DataFrame({"a": [[1]], "b": [[1]]})
    try:
        df.select(pl.col("a").list.similarity("b").alias("similarity"))
    except Exception:
        return False
    return True


def test_python_sequence_similarity_core_cases() -> None:
    assert _score([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert _score([1, 2, 3], [4, 5, 6]) == pytest.approx(0.0)
    assert _score([], []) == pytest.approx(1.0)
    assert _score([], [1, 2]) == pytest.approx(0.0)

    longer_tail = _score([1, 2, 3], [1, 2, 3, 4, 5])
    assert longer_tail == pytest.approx((1.0 + 0.9 + 0.81) / (1.0 + 0.9 + 0.81 + 0.729 + 0.6561))


def test_python_sequence_similarity_is_symmetric_and_position_aware() -> None:
    left = _score([1, 2, 3], [2, 1, 3])
    right = _score([2, 1, 3], [1, 2, 3])

    assert left == pytest.approx(right)
    assert 0.0 < left < 1.0


def test_python_sequence_similarity_p_one_matches_set_jaccard() -> None:
    assert _score([1, 2, 3], [2, 3, 4, 5], p=1.0) == pytest.approx(2 / 5)


def test_python_sequence_similarity_uses_best_duplicate_position() -> None:
    assert _score([1, 2, 1], [1, 2]) == pytest.approx(1.0)
    assert _score(["a", "b", "a"], ["a", "b"]) == pytest.approx(1.0)


def test_python_sequence_similarity_ignores_null_document_ids() -> None:
    assert _score([None], []) == pytest.approx(1.0)
    assert _score([1, None], [1]) == pytest.approx(1.0)


def test_python_sequence_similarity_accepts_primitive_scalars() -> None:
    assert _score([True, False], [False, True]) == pytest.approx(_two_item_swapped_score())
    assert _score([1.5, 2.5, 3.5], [2.5, 1.5, 3.5]) == pytest.approx(_swapped_score())
    assert _score([b"a", b"b"], [b"a", b"b"]) == pytest.approx(1.0)
    assert _score([date(2024, 1, 1)], [date(2024, 1, 1)]) == pytest.approx(1.0)


def test_python_sequence_similarity_validation_errors() -> None:
    with pytest.raises(TypeError, match="must not mix primitive scalar types"):
        py_list_similarity([1], ["1"])

    with pytest.raises(TypeError, match="primitive scalar values"):
        py_list_similarity([[1]], [[1]])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="0 < p <= 1"):
        py_list_similarity([1], [1], p=0)

    with pytest.raises(ValueError, match="0 < p <= 1"):
        py_list_similarity([1], [1], p=float("nan"))

    with pytest.raises(TypeError, match="real number"):
        py_list_similarity([1], [1], p="0.9")  # type: ignore[arg-type]


def test_polars_list_similarity_rejects_python_sequences() -> None:
    with pytest.raises(TypeError, match="use py_list_similarity"):
        pl.col("a").list.similarity([1])  # type: ignore[arg-type]


def test_polars_python_fallback_with_int_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_list_similarity, "_native_library_available", lambda: False)
    df = pl.DataFrame(
        {
            "a": [[1, 2, 3], [1, 2, 3], [1, None], None, []],
            "b": [[1, 2, 3], [2, 1, 3], [4], [1], []],
        }
    )

    out = df.with_columns(pl.col("a").list.similarity("b").alias("similarity"))

    values = out["similarity"].to_list()
    assert values[:3] == pytest.approx(
        [
            1.0,
            (0.9 + 0.9 + 0.81) / (1.0 + 1.0 + 0.81),
            0.0,
        ]
    )
    assert values[3] is None
    assert values[4] == pytest.approx(1.0)


def test_polars_python_fallback_with_string_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_list_similarity, "_native_library_available", lambda: False)
    df = pl.DataFrame(
        {
            "a": [["a", "b", "c"], ["x"]],
            "b": [["b", "a", "c"], ["y"]],
        }
    )

    out = df.select(pl.col("a").list.similarity("b").alias("similarity"))

    assert out["similarity"].to_list() == pytest.approx(
        [
            _swapped_score(),
            0.0,
        ]
    )


@pytest.mark.skipif(
    not _native_list_similarity_available(),
    reason="native list_similarity plugin has not been built",
)
def test_native_list_similarity_plugin_matches_python() -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 2, 3], [1, 2, 3], [], [None]],
            "b": [[1, 2, 3], [2, 1, 3], [1], []],
        },
        schema={"a": pl.List(pl.Int64), "b": pl.List(pl.Int64)},
    )

    out = df.select(pl.col("a").list.similarity("b").alias("similarity"))

    assert out["similarity"].to_list() == pytest.approx(
        [
            1.0,
            _swapped_score(),
            0.0,
            1.0,
        ]
    )


@pytest.mark.skipif(
    not _native_list_similarity_available(),
    reason="native list_similarity plugin has not been built",
)
def test_native_list_similarity_plugin_accepts_string_lists() -> None:
    df = pl.DataFrame(
        {
            "a": [["a", "b", "c"], ["x"]],
            "b": [["b", "a", "c"], ["y"]],
        }
    )

    out = df.select(pl.col("a").list.similarity("b").alias("similarity"))

    assert out["similarity"].to_list() == pytest.approx(
        [
            _swapped_score(),
            0.0,
        ]
    )


@pytest.mark.skipif(
    not _native_list_similarity_available(),
    reason="native list_similarity plugin has not been built",
)
@pytest.mark.parametrize(
    ("dtype", "left", "right", "expected"),
    [
        (pl.Int32, [[1, 2, 3]], [[2, 1, 3]], _swapped_score()),
        (pl.UInt32, [[1, 2, 3]], [[2, 1, 3]], _swapped_score()),
        (pl.Float64, [[1.5, 2.5, 3.5]], [[2.5, 1.5, 3.5]], _swapped_score()),
        (pl.Boolean, [[True, False]], [[False, True]], _two_item_swapped_score()),
        (pl.Binary, [[b"a", b"b", b"c"]], [[b"b", b"a", b"c"]], _swapped_score()),
        (
            pl.Date,
            [[date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]],
            [[date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 3)]],
            _swapped_score(),
        ),
    ],
)
def test_native_list_similarity_plugin_accepts_primitive_lists(
    dtype: Any,
    left: list[list[object]],
    right: list[list[object]],
    expected: float,
) -> None:
    df = pl.DataFrame(
        {"a": left, "b": right},
        schema={"a": pl.List(dtype), "b": pl.List(dtype)},
    )

    out = df.select(pl.col("a").list.similarity("b").alias("similarity"))

    assert out["similarity"].to_list() == pytest.approx([expected])


@pytest.mark.skipif(
    not _native_list_similarity_available(),
    reason="native list_similarity plugin has not been built",
)
def test_native_list_similarity_plugin_rejects_mixed_inner_dtypes() -> None:
    df = pl.DataFrame({"ints": [[1]], "strings": [["1"]]})

    with pytest.raises(pl.exceptions.ComputeError, match="must match exactly"):
        df.select(pl.col("ints").list.similarity("strings").alias("similarity"))

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from itertools import permutations
from random import Random
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_list_math import (
    _list_kendall_tau,
    install,
    list_kendall_tau,
    list_mean_kendall_tau,
    list_mean_kendall_tau_to,
    py_list_kendall_tau,
)


@pytest.fixture(params=["fallback", "native"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "native":
        if not _list_kendall_tau._native_library_available():
            pytest.skip("native extension has not been built")
    else:
        monkeypatch.setattr(_list_kendall_tau, "_native_library_available", lambda: False)
    return request.param


def _oracle(left: list[int | None], right: list[int | None]) -> float | None:
    a = list(dict.fromkeys(value for value in left if value is not None))
    b = list(dict.fromkeys(value for value in right if value is not None))
    common = [value for value in a if value in b]
    scores = [
        1 if b.index(common[i]) < b.index(common[j]) else -1
        for i in range(len(common))
        for j in range(i + 1, len(common))
    ]
    return sum(scores) / len(scores) if scores else None


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [3, 2, 1], -1.0),
        ([1, 2, 3], [2, 1, 3], 1 / 3),
        ([1, 2, 3, 4], [2, 3, 4, 1], 0.0),
        ([99, 1, None, 2, 1, 3], [3, 2, 88, 1, 3], -1.0),
        ([], [], None),
        ([None], [], None),
        ([1], [1], None),
        ([1, 2], [2, 3], None),
        ([1, 2], [3, 4], None),
    ],
)
def test_python_scores(left: Any, right: Any, expected: float | None, normalize: bool) -> None:
    if expected is not None and normalize:
        expected = (expected + 1) / 2
    result = py_list_kendall_tau(left, right, normalize=normalize)
    reverse = py_list_kendall_tau(right, left, normalize=normalize)
    if expected is None:
        assert result is None and reverse is None
    else:
        assert result == pytest.approx(expected)
        assert reverse == pytest.approx(expected)


def test_permutations_and_partial_rankings_against_pairwise_oracle(backend: str) -> None:
    orders = list(permutations(range(4)))
    pairs: list[tuple[list[int | None], list[int | None]]] = [
        (list(a), list(b)) for a in orders for b in orders
    ]
    rng = Random(42)
    pairs.extend(
        (
            [rng.choice([None, *range(8)]) for _ in range(rng.randrange(15))],
            [rng.choice([None, *range(8)]) for _ in range(rng.randrange(15))],
        )
        for _ in range(200)
    )
    expected = [_oracle(a, b) for a, b in pairs]
    for (a, b), score in zip(pairs, expected, strict=True):
        result = py_list_kendall_tau(a, b)
        assert result is None if score is None else result == pytest.approx(score)
    frame = pl.DataFrame({"a": [a for a, _ in pairs], "b": [b for _, b in pairs]})
    actual = frame.select(list_kendall_tau("a", "b").alias("tau"))
    assert_frame_equal(actual, pl.DataFrame({"tau": expected}, schema={"tau": pl.Float64}))


@pytest.mark.parametrize("normalize", [False, True])
def test_nulls_broadcasting_chunks_lazy_and_streaming(backend: str, normalize: bool) -> None:
    frame = pl.DataFrame({"a": [[1, 2, 3], [3, 2, 1], None, [], [1, 1, None]]})
    frame = pl.concat([frame[:2], frame[2:]], rechunk=False)
    expected = pl.DataFrame({"a": [1.0, 0.0 if normalize else -1.0, None, None, None]})
    expression = list_kendall_tau("a", pl.lit([1, 2, 3]), normalize=normalize)
    for engine in ("auto", "streaming"):
        actual = frame.lazy().select(expression).collect(engine=engine)
        assert_frame_equal(actual, expected)
    reverse = frame.select(list_kendall_tau(pl.lit([1, 2, 3]), "a", normalize=normalize))
    assert reverse.to_series().to_list() == expected.to_series().to_list()
    empty = frame.head(0).select(expression)
    assert empty.shape == (0, 1)
    assert empty.dtypes == [pl.Float64]


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (pl.Int8, [1, 2, 3]),
        (pl.Int32, [1, 2, 3]),
        (pl.Int64, [2**60, 2**60 + 1, 2**60 + 2]),
        (pl.Int128, [2**100, 2**100 + 1, 2**100 + 2]),
        (pl.UInt64, [2**63, 2**63 + 1, 2**63 + 2]),
        (pl.Float32, [1.5, 2.5, 3.5]),
        (pl.Float64, [-0.0, 0.0, float("nan")]),
        (pl.Boolean, [True, False]),
        (pl.String, ["a", "б", "c"]),
        (pl.Binary, [b"a", b"b", b"c"]),
        (pl.Date, [date(2024, 1, i) for i in range(1, 4)]),
        (pl.Datetime("us"), [datetime(2024, 1, i) for i in range(1, 4)]),
        (pl.Duration("us"), [timedelta(seconds=i) for i in range(3)]),
        (pl.Time, [time(i) for i in range(3)]),
    ],
)
def test_primitive_ids(backend: str, dtype: Any, values: list[Any]) -> None:
    frame = pl.DataFrame(
        {"a": [values], "b": [values[::-1]]},
        schema={"a": pl.List(dtype), "b": pl.List(dtype)},
    )
    assert frame.select(list_kendall_tau("a", "b")).item() == -1.0
    assert py_list_kendall_tau(values, values[::-1]) == -1.0


def test_arrays_and_null_inner_dtype(backend: str) -> None:
    frame = pl.DataFrame(
        {"a": [[1, 2, 3]], "b": [[3, 2, 1]]},
        schema={"a": pl.Array(pl.Int64, 3), "b": pl.List(pl.Int64)},
    )
    assert frame.select(list_kendall_tau("a", "b")).item() == -1.0
    frame = pl.DataFrame({"a": [[None, None]], "b": [[1, 2]]})
    assert frame.select(list_kendall_tau("a", "b")).item() is None


@pytest.mark.parametrize("normalize", [False, True])
def test_nested_means_skip_null_and_undefined_pairs(backend: str, normalize: bool) -> None:
    frame = pl.DataFrame(
        {
            "a": [[[1, 2, 3], [2, 1, 3], [3, 2, 1], [9], None], [], None, [[1, 2]]],
            "b": [[[1, 2, 3], [3, 2, 1], None, []], [], [[1, 2]], None],
        },
        schema={"a": pl.List(pl.List(pl.Int64)), "b": pl.List(pl.List(pl.Int64))},
    )
    self_scores = [-1 / 3, 0.0, -2 / 3]
    cross_scores = [0.0, 0.0, 0.0]
    if normalize:
        self_scores = [(value + 1) / 2 for value in self_scores]
        cross_scores = [0.5] * 3
    actual = (
        frame.lazy()
        .select(
            list_mean_kendall_tau("a", normalize=normalize).alias("self"),
            list_mean_kendall_tau_to("a", "b", normalize=normalize).alias("cross"),
        )
        .collect()
    )
    expected = pl.DataFrame(
        {
            "self": [[*self_scores, None, None], [], None, [None]],
            "cross": [[*cross_scores, None, None], [], None, [None]],
        },
        schema={"self": pl.List(pl.Float64), "cross": pl.List(pl.Float64)},
    )
    assert_frame_equal(actual, expected)


def test_nested_arrays_and_reference_broadcast(backend: str) -> None:
    frame = pl.DataFrame(
        {"a": [[[1, 2], [2, 1]], [[2, 1], [1, 2]]]},
        schema={"a": pl.Array(pl.Array(pl.Int64, 2), 2)},
    )
    result = frame.select(list_mean_kendall_tau("a"))
    assert result.to_series().to_list() == [[-1.0, -1.0], [-1.0, -1.0]]
    result = frame.select(list_mean_kendall_tau_to("a", pl.lit([[1, 2]])))
    assert result.to_series().to_list() == [[1.0, -1.0], [-1.0, 1.0]]


def test_runtime_namespace_and_install(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame({"a": [[1, 2]], "b": [[2, 1]], "nested": [[[1, 2], [2, 1]]]})
    assert frame.select(pl.col("a").list.kendall_tau("b", normalize=True)).item() == 0.0
    assert_frame_equal(
        frame.select(pl.col("nested").list.mean_kendall_tau()),
        frame.select(list_mean_kendall_tau("nested")),
    )
    assert_frame_equal(
        frame.select(pl.col("nested").list.mean_kendall_tau_to("nested")),
        frame.select(list_mean_kendall_tau_to("nested", "nested")),
    )
    namespace: Any = type(pl.col("a").list)
    sentinel = object()
    monkeypatch.setattr(namespace, "kendall_tau", sentinel)
    install()
    assert namespace.kendall_tau is sentinel
    install(overwrite=True)
    assert namespace.kendall_tau is not sentinel


@pytest.mark.parametrize("dtype", [pl.Datetime("ns"), pl.Duration("ns"), pl.Time])
def test_nanosecond_ids_preserve_precision(backend: str, dtype: Any) -> None:
    frame = pl.DataFrame(
        {"a": [[1, 2, 3]], "b": [[3, 2, 1]]},
        schema={"a": pl.List(dtype), "b": pl.List(dtype)},
    )
    assert frame.select(list_kendall_tau("a", "b")).item() == -1.0
    nested = pl.DataFrame({"a": [[[1, 2, 3], [3, 2, 1]]]}, schema={"a": pl.List(pl.List(dtype))})
    assert nested.select(list_mean_kendall_tau("a")).to_series().to_list() == [[-1.0, -1.0]]


def test_validation(backend: str) -> None:
    with pytest.raises(TypeError, match="use py_list_kendall_tau"):
        list_kendall_tau("a", [1, 2])  # type: ignore[arg-type]
    for value in (1, "yes", None):
        with pytest.raises(TypeError, match="normalize must be a bool"):
            list_kendall_tau("a", "b", normalize=value)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="normalize must be a bool"):
            py_list_kendall_tau([1], [1], normalize=value)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must not mix primitive scalar types"):
        py_list_kendall_tau([1, 2], ["1", "2"])
    with pytest.raises(TypeError, match="primitive scalar"):
        py_list_kendall_tau([[1]], [[1]])  # type: ignore[list-item]
    frame = pl.DataFrame({"a": [[1, 2]], "b": [[1, 2]]}).with_columns(
        pl.col("b").cast(pl.List(pl.Int32))
    )
    with pytest.raises((TypeError, pl.exceptions.PolarsError), match="must match exactly"):
        frame.select(list_kendall_tau("a", "b"))
    with pytest.raises((TypeError, pl.exceptions.PolarsError), match="List or Array"):
        pl.DataFrame({"a": [1], "b": [2]}).select(list_kendall_tau("a", "b"))
    with pytest.raises((TypeError, pl.exceptions.PolarsError), match="primitive list inner dtype"):
        pl.DataFrame({"a": [[{"x": 1}]]}).select(list_kendall_tau("a", "a"))

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_list_math import (
    _list_containment,
    install,
    list_containment,
    list_mean_containment,
    list_mean_containment_to,
    py_list_containment,
)


@pytest.fixture(params=[False, True], ids=["fallback", "native"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param and not _list_containment._native_library_available():
        pytest.skip("native extension is not built; run make develop")
    monkeypatch.setattr(_list_containment, "_native_library_available", lambda: request.param)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1, 2], [1, 2, 3], 1.0),
        ([1, 2, 3], [1, 2], 2 / 3),
        ([1, 1, 2, None], [1, 1, None], 0.5),
        ([2, 1], [1, 2], 1.0),
        ([1], [2], 0.0),
        ([], [], 0.0),
        ([], [1], 0.0),
        ([1], [], 0.0),
        ([None], [None], 0.0),
    ],
)
def test_python_containment(left: Any, right: Any, expected: float) -> None:
    assert py_list_containment(left, right) == pytest.approx(expected)


def test_python_containment_validation() -> None:
    for left, right in [([1], ["1"]), ([True, 1], [True])]:
        with pytest.raises(TypeError, match="must not mix"):
            py_list_containment(left, right)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="primitive scalar"):
        py_list_containment([[1]], [[1]])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="use py_list_containment"):
        list_containment([1], "b")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Polars expression"):
        list_mean_containment([[]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Polars expression"):
        list_mean_containment_to("a", [[]])  # type: ignore[arg-type]


def test_pairwise(backend: None) -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 2], [1, 2, 3], [1, 1, 2, None], [], [None], None, [1]],
            "b": [[1, 2, 3], [2, 1], [1, 1], [], [None], [1], None],
        },
        schema={"a": pl.List(pl.Int64), "b": pl.List(pl.Int64)},
    )
    result = df.lazy().select(list_containment("a", "b").alias("score")).collect()
    assert result.schema == {"score": pl.Float64}
    assert result["score"].to_list() == pytest.approx([1, 2 / 3, 0.5, 0, 0, None, None])
    assert_frame_equal(result, df.select(pl.col("a").list.containment("b").alias("score")))


@pytest.mark.parametrize(
    ("dtype", "items"),
    [
        (pl.String, ["a", "b"]),
        (pl.Int32, [1, 2]),
        (pl.UInt64, [1, 2]),
        (pl.Float32, [1.5, 2.5]),
        (pl.Float64, [float("nan"), float("inf")]),
        (pl.Boolean, [True, False]),
        (pl.Binary, [b"a", b"b"]),
        (pl.Date, [date(2026, 1, 1), date(2026, 1, 2)]),
    ],
)
def test_primitive_types(backend: None, dtype: Any, items: Any) -> None:
    df = pl.DataFrame(
        {"a": [items], "b": [[items[0]]]}, schema={"a": pl.List(dtype), "b": pl.List(dtype)}
    )
    assert df.select(list_containment("a", "b")).item() == 0.5
    nested = pl.DataFrame({"a": [[items, [items[0]]]]}, schema={"a": pl.List(pl.List(dtype))})
    assert nested.select(list_mean_containment("a")).to_series().to_list() == [[0.5, 1.0]]


def test_mean_self(backend: None) -> None:
    df = pl.DataFrame(
        {"groups": [[[1, 2], [1, 2, 3], [4]], [[1], None, [1, 2]], [[]], [[], []], [], None]},
        schema={"groups": pl.List(pl.List(pl.Int64))},
    )
    result = df.select(list_mean_containment("groups"))
    assert result.schema == {"groups": pl.List(pl.Float64)}
    values = result.to_series().to_list()
    assert values[0] == pytest.approx([0.5, 1 / 3, 0.0])
    assert values[1:] == [[1.0, None, 0.5], [None], [0.0, 0.0], [], None]
    assert_frame_equal(result, df.select(pl.col("groups").list.mean_containment()))


def test_mean_to(backend: None) -> None:
    df = pl.DataFrame(
        {
            "a": [[[1, 2], [1, 2, 3], None, []], [[1]], [[1], []], [[1]], [], None],
            "b": [[[1, 2], [2], None], [], None, [None], [[1]], [[1]]],
        },
        schema={"a": pl.List(pl.List(pl.Int64)), "b": pl.List(pl.List(pl.Int64))},
    )
    result = df.select(list_mean_containment_to("a", "b"))
    values = result.to_series().to_list()
    assert values[0] == pytest.approx([0.75, 0.5, None, 0.0])
    assert values[1:] == [[None], [None, None], [None], [], None]
    assert_frame_equal(result, df.select(pl.col("a").list.mean_containment_to("b")))
    # Unlike self means, the reference variant includes equal lists and self matches.
    same = pl.DataFrame({"a": [[[1], [1, 2]]]})
    assert same.select(list_mean_containment_to("a", "a")).to_series().to_list() == [[1.0, 0.75]]


def test_broadcast_slices_chunks_and_empty_frames(backend: None) -> None:
    df = pl.concat([pl.DataFrame({"a": [[1], [1, 2]]})] * 3, rechunk=False).slice(1, 4)
    assert df.select(list_containment("a", pl.lit([1]))).to_series().to_list() == [0.5, 1, 0.5, 1]
    assert df.select(list_containment(pl.lit([1]), "a")).to_series().to_list() == [1] * 4
    nested = pl.concat([pl.DataFrame({"a": [[[1], [1, 2]]]})] * 3, rechunk=False).slice(1, 2)
    assert nested.select(list_mean_containment("a")).to_series().to_list() == [[1, 0.5]] * 2
    assert (
        nested.select(list_mean_containment_to("a", pl.lit([[1]]))).to_series().to_list()
        == [[1, 0.5]] * 2
    )
    assert df.head(0).select(list_containment("a", "a")).shape == (0, 1)
    assert nested.head(0).select(list_mean_containment("a")).shape == (0, 1)
    assert nested.head(0).select(list_mean_containment_to("a", "a")).shape == (0, 1)


def test_null_dtype_and_invalid_types(backend: None) -> None:
    df = pl.DataFrame({"a": [[None], []], "b": [[1], []]})
    assert df.select(list_containment("a", "b")).to_series().to_list() == [0.0, 0.0]
    nested = pl.DataFrame({"a": [[[None], []]]})
    assert nested.select(list_mean_containment("a")).to_series().to_list() == [[0.0, 0.0]]
    for data in ({"a": [[1]], "b": [["1"]]}, {"a": [[{"id": 1}]], "b": [[{"id": 1}]]}):
        with pytest.raises((TypeError, pl.exceptions.PolarsError)):
            pl.DataFrame(data).select(list_containment("a", "b"))
    with pytest.raises((TypeError, pl.exceptions.PolarsError)):
        pl.DataFrame({"a": [1]}).select(list_mean_containment("a"))


def test_namespace_install_preserves_existing_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace: Any = type(pl.col("a").list)
    sentinel = object()
    monkeypatch.setattr(namespace, "containment", sentinel)
    install()
    assert namespace.containment is sentinel
    install(overwrite=True)
    assert callable(namespace.containment)

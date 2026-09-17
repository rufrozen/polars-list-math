from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from polars_list_math import ZipBuilder, _list_zip, list_zip


@pytest.fixture(params=["native", "fallback"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param == "native":
        if not _list_zip._native_library_available():
            pytest.skip("native zip is not built")
    else:
        monkeypatch.setattr(_list_zip, "_native_library_available", lambda: False)


def test_outer_level_matches_zip_and_build_is_reusable(backend: None) -> None:
    frame = pl.DataFrame({"a": [[1, 2], []], "b": [[10], [30]]})
    builder = ZipBuilder("a", "b", pad=True, fields=["left", "right"])
    before = builder.build("z")
    expected = frame.select(list_zip("a", "b", pad=True, fields=["left", "right"]))
    assert frame.select(before).to_series().to_list() == expected.to_series().to_list()
    assert frame.select(builder.build()).columns == ["left"]
    assert frame.select(builder.build("z")).equals(frame.select(before))
    builder.filter(pl.element().struct.field("left").is_not_null())
    assert frame.select(builder.build("z")).to_series().to_list() == [
        [{"left": 1, "right": 10}, {"left": 2, "right": None}],
        [],
    ]
    # An expression already returned by build is unaffected by later mutations.
    assert frame.select(before).to_series().to_list() == expected.to_series().to_list()


def test_operations_at_each_level_run_in_order(backend: None) -> None:
    frame = pl.DataFrame({"a": [[[1, 2], [3], [4, 5]]], "b": [[[10, 20], [30], [40, 50]]]})
    element = pl.element()
    expr = (
        ZipBuilder("a", "b")
        .filter(element.struct.field("a").list.len() > 1)
        .eval(
            pl.struct(
                element.struct.field("a").list.reverse().alias("a"),
                element.struct.field("b").alias("b"),
            )
        )
        .level(fields=["x", "y"])
        .filter(element.struct.field("x") % 2 == 0)
        .eval(element.struct.field("x") + element.struct.field("y"), parallel=True)
        .filter(element > 20)
        .eval(element * 2)
        .build("result")
    )
    assert frame.lazy().select(expr).collect()["result"].to_list() == [[[], [108]]]


@pytest.mark.parametrize("depth", [1, 2, 3, 6])
def test_arbitrary_nesting_preserves_shape(backend: None, depth: int) -> None:
    left: Any = [1, 2]
    right: Any = [10, 20]
    expected: Any = [11, 22]
    builder = ZipBuilder("a", "b")
    for _ in range(depth - 1):
        left, right, expected = [left, left], [right, right], [expected, expected]
        builder.level()
    expr = builder.eval(pl.element().struct.field("a") + pl.element().struct.field("b")).build()
    frame = pl.DataFrame({"a": [left], "b": [right]})
    assert frame.select(expr).to_series().to_list() == [expected]


@pytest.mark.parametrize("outer_pad", [False, True])
@pytest.mark.parametrize("inner_pad", [False, True])
def test_independent_padding_and_null_lists(
    backend: None, outer_pad: bool, inner_pad: bool
) -> None:
    frame = pl.DataFrame(
        {"a": [[[1], [2]], None, [], [[None]], [None]], "b": [[[10, 20]], [[3]], [], [[4]], [[5]]]},
        schema={"a": pl.List(pl.List(pl.Int64)), "b": pl.List(pl.List(pl.Int64))},
    )
    result = frame.select(ZipBuilder("a", "b", pad=outer_pad).level(pad=inner_pad).build("z"))[
        "z"
    ].to_list()
    first: list[Any] = [[{"a": 1, "b": 10}]]
    if inner_pad:
        first[0].append({"a": None, "b": 20})
    if outer_pad:
        # Outer padding inserts a null list, which propagates through the inner zip.
        first.append(None)
    assert result == [first, None, [], [[{"a": None, "b": 4}]], [None]]


def test_scalar_broadcast_at_multiple_depths(backend: None) -> None:
    frame = pl.DataFrame({"a": [[[1, 2], [3]], []], "currency": ["RUB", None]})
    expr = (
        ZipBuilder("a", "currency", pl.lit(True), broadcast=True, fields=["item", "unit", "ok"])
        .level(broadcast=True, fields=["value", "currency", "enabled"])
        .build("z")
    )
    assert frame.select(expr)["z"].to_list() == [
        [
            [{"value": value, "currency": "RUB", "enabled": True} for value in [1, 2]],
            [{"value": 3, "currency": "RUB", "enabled": True}],
        ],
        [],
    ]
    with pytest.raises((TypeError, pl.exceptions.PolarsError)):
        frame.select(ZipBuilder("a", "currency", broadcast=True).level().build())


@pytest.mark.parametrize("values", [[], [[]], [None]])
def test_empty_and_all_null_input(backend: None, values: Any) -> None:
    frame = pl.DataFrame(
        {"a": values, "b": values},
        schema={"a": pl.List(pl.List(pl.Int64)), "b": pl.List(pl.List(pl.Int64))},
    )
    expr = ZipBuilder("a", "b").level().build("z")
    assert frame.lazy().select(expr).collect()["z"].to_list() == values


def test_expression_inputs_and_three_columns(backend: None) -> None:
    frame = pl.DataFrame({"a": [[[1, 2]]], "b": [[[10, 20]]], "c": [[[100, 200]]]})
    builder = ZipBuilder(pl.col("a").alias("first"), "b", "c").level()
    expr = builder.eval(
        pl.element().struct.field("first")
        + pl.element().struct.field("b")
        + pl.element().struct.field("c")
    ).build("total")
    assert frame.select(expr)["total"].to_list() == [[[111, 222]]]


def test_validation_and_failed_calls_do_not_mutate_builder() -> None:
    with pytest.raises(ValueError, match="at least two"):
        ZipBuilder("a")
    with pytest.raises(ValueError, match="unique"):
        ZipBuilder("a", "a")
    with pytest.raises(TypeError, match="single string"):
        ZipBuilder("a", "b", fields="ab")
    builder = ZipBuilder("a", "b")
    with pytest.raises(ValueError, match="exactly 2"):
        builder.level(fields=["a"])
    with pytest.raises(TypeError, match="pad must"):
        builder.level(pad=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="broadcast must"):
        ZipBuilder("a", "b", broadcast="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Polars expression"):
        builder.eval("a")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Polars expression"):
        builder.filter(True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parallel must"):
        builder.eval(pl.element(), parallel=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pl.element"):
        builder.filter(pl.col("a") > 0)
    with pytest.raises(TypeError, match="name must"):
        builder.build(123)  # type: ignore[arg-type]
    assert pl.DataFrame({"a": [[1]], "b": [[2]]}).select(builder.build()).to_series().to_list() == [
        [{"a": 1, "b": 2}]
    ]

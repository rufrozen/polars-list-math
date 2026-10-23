from collections.abc import Iterator

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_list_math import apply_schema, clean_schema, dfs_to_lazy_df, find_null_schema


def test_finds_null_columns_and_fields_from_user_examples() -> None:
    frame = pl.LazyFrame(
        {
            "null": [None, None],
            "lists": [[[None]], None],
            "partial": [[{"a": 1, "b": None, "c": "value"}], [{"a": None, "b": None, "c": None}]],
            "all_null": [[[{"a": None, "b": None}]], []],
        },
        schema={
            "null": pl.Null,
            "lists": pl.List(pl.List(pl.Null)),
            "partial": pl.List(pl.Struct({"a": pl.Int64, "b": pl.Null, "c": pl.String})),
            "all_null": pl.List(pl.List(pl.Struct({"a": pl.Null, "b": pl.Null}))),
        },
    )

    assert find_null_schema(frame) == pl.Schema(
        {
            "null": pl.Null,
            "lists": pl.List(pl.List(pl.Null)),
            "partial": pl.List(pl.Struct({"b": pl.Null})),
            "all_null": pl.List(pl.List(pl.Struct({"a": pl.Null, "b": pl.Null}))),
        }
    )


def test_checks_values_of_typed_leaves_and_keeps_falsey_values() -> None:
    frame = pl.LazyFrame(
        {
            "absent": [None, None],
            "zero": [None, 0],
            "false": [None, False],
            "empty_string": [None, ""],
            "nan": [None, float("nan")],
            "nested": [None, [{"missing": [None, None], "present": [None, 7]}]],
        },
        schema={
            "absent": pl.Date,
            "zero": pl.Int64,
            "false": pl.Boolean,
            "empty_string": pl.String,
            "nan": pl.Float64,
            "nested": pl.List(
                pl.Struct({"missing": pl.List(pl.Float64), "present": pl.List(pl.Int64)})
            ),
        },
    )

    assert find_null_schema(frame) == pl.Schema(
        {"absent": pl.Null, "nested": pl.List(pl.Struct({"missing": pl.List(pl.Null)}))}
    )


@pytest.mark.parametrize("depth", [0, 1, 2, 8, 20])
def test_alternating_containers_at_arbitrary_depth(depth: int) -> None:
    dtype: pl.DataType = pl.Struct({"a": pl.Int64, "b": pl.String})
    mask: pl.DataType = pl.Struct({"b": pl.Null})
    cleaned: pl.DataType = pl.Struct({"a": pl.Int64})
    value: object = {"a": 42, "b": None}
    expected: object = {"a": 42}
    for index in range(depth):
        if index % 2:
            dtype, mask = pl.Struct({"node": dtype}), pl.Struct({"node": mask})
            value = {"node": value}
            cleaned, expected = pl.Struct({"node": cleaned}), {"node": expected}
        else:
            dtype, mask = pl.List(dtype), pl.List(mask)
            value = [None, value, None]
            cleaned, expected = pl.List(cleaned), [None, expected, None]

    frame = pl.LazyFrame({"x": [None, value]}, schema={"x": dtype})
    assert find_null_schema(frame) == pl.Schema({"x": mask})
    target = clean_schema(frame.collect_schema(), pl.Schema({"x": mask}))
    assert target == pl.Schema({"x": cleaned})
    assert_frame_equal(
        apply_schema(frame, target).collect(),
        pl.DataFrame({"x": [None, expected]}, schema={"x": cleaned}),
    )


@pytest.mark.parametrize("values", [[], [None], [[]], [[None]], [[[]]], [[[None]]]])
def test_empty_or_null_containers_have_no_values(values: list[object]) -> None:
    frame = pl.LazyFrame({"x": values}, schema={"x": pl.List(pl.List(pl.Int64))})
    assert find_null_schema(frame) == pl.Schema({"x": pl.List(pl.List(pl.Null))})


def test_null_struct_parents_and_empty_structs() -> None:
    frame = pl.LazyFrame(
        {"null_parent": [None], "empty": [{}]},
        schema={
            "null_parent": pl.Struct({"a": pl.Int64, "b": pl.List(pl.String)}),
            "empty": pl.Struct({}),
        },
    )
    assert find_null_schema(frame) == pl.Schema(
        {"null_parent": pl.Struct({"a": pl.Null, "b": pl.List(pl.Null)}), "empty": pl.Struct({})}
    )


def test_empty_frame_and_schema() -> None:
    assert find_null_schema(pl.LazyFrame()) == pl.Schema()
    assert find_null_schema(pl.LazyFrame(schema={"a": pl.Int64, "b": pl.String})) == pl.Schema(
        {"a": pl.Null, "b": pl.Null}
    )


def test_multidimensional_arrays() -> None:
    dtype = pl.Array(pl.Struct({"a": pl.Int64, "b": pl.String}), (2, 2))
    frame = pl.LazyFrame(
        {"x": [None, [[None, {"a": 1, "b": None}], [None, None]]]}, schema={"x": dtype}
    )
    assert find_null_schema(frame) == pl.Schema({"x": pl.Array(pl.Struct({"b": pl.Null}), (2, 2))})


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_scans_all_batches_once_and_respects_filters(
    monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    monkeypatch.setenv("POLARS_ENGINE_AFFINITY", engine)
    scans: list[int] = []
    schema = pl.Schema({"a": pl.Int64, "b": pl.Int64})

    class Batches:
        def __iter__(self) -> Iterator[pl.DataFrame]:
            scans.append(1)
            yield pl.DataFrame({"a": [None], "b": [None]}, schema=schema)
            yield pl.DataFrame({"a": [1], "b": [None]}, schema=schema)

    frame = dfs_to_lazy_df(Batches(), schema)
    assert find_null_schema(frame) == pl.Schema({"b": pl.Null})
    assert len(scans) == 1
    assert find_null_schema(frame.filter(pl.col("a").is_null())) == pl.Schema(
        {"a": pl.Null, "b": pl.Null}
    )
    assert len(scans) == 2
    assert find_null_schema(frame.filter(pl.col("a") > 10)) == pl.Schema(
        {"a": pl.Null, "b": pl.Null}
    )
    assert len(scans) == 3


def test_names_are_literal() -> None:
    frame = pl.LazyFrame(
        {"*": [None], "^x$": [{"*": None, "a.b": 1}], "x": [1]},
        schema={"*": pl.Int64, "^x$": pl.Struct({"*": pl.String, "a.b": pl.Int64}), "x": pl.Int64},
    )
    assert find_null_schema(frame) == pl.Schema({"*": pl.Null, "^x$": pl.Struct({"*": pl.Null})})


def test_known_null_schema_does_not_execute_source() -> None:
    schema = pl.Schema({"x": pl.List(pl.Struct({"empty": pl.Struct({}), "absent": pl.Null}))})

    class Batches:
        def __iter__(self) -> Iterator[pl.DataFrame]:
            raise AssertionError("A statically absent schema must not execute the source")

    assert find_null_schema(dfs_to_lazy_df(Batches(), schema)) == schema


@pytest.mark.parametrize("size", [0, 2])
def test_arrays_without_values(size: int) -> None:
    frame = pl.LazyFrame({"x": [None, [None] * size]}, schema={"x": pl.Array(pl.Int64, size)})
    mask = find_null_schema(frame)
    assert mask == pl.Schema({"x": pl.Array(pl.Null, size)})
    assert clean_schema(frame.collect_schema(), mask) == pl.Schema()

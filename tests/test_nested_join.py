from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_list_math import _nested_join, nested_join, nested_lookup


@pytest.fixture(params=["native", "fallback"])
def backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "native":
        if not _nested_join._native_library_available():
            pytest.skip("build native plugin with make develop")
    else:
        monkeypatch.setattr(_nested_join, "_native_library_available", lambda: False)
    return request.param


def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    left = pl.DataFrame(
        {
            "query": ["b", "a", "b", "missing", "empty"],
            "sku": pl.Series(
                [[[30, 10, 30], [], [20]], [[30]], None, [[10], None, []], []],
                dtype=pl.List(pl.List(pl.Int64)),
            ),
            "unchanged": [9, 8, 7, 6, 5],
        }
    )
    right = pl.DataFrame(
        {"q": ["a", "b", "b"], "id": [30, 10, 30], "grade": [8, 1, 3]},
        schema_overrides={"grade": pl.Int8},
    )
    return left, right


def join(left: pl.DataFrame | pl.LazyFrame, right: pl.DataFrame | pl.LazyFrame, **kwargs) -> Any:
    return nested_join(
        left,
        right,
        on={"query": "q", "sku": "id"},
        shape="sku",
        values={"grade": "label"},
        **kwargs,
    )


@pytest.mark.parametrize(
    "left_lazy,right_lazy", [(False, False), (True, False), (False, True), (True, True)]
)
def test_join_preserves_order_shape_and_return_type(backend, left_lazy, right_lazy):
    left, right = frames()
    result = join(left.lazy() if left_lazy else left, right.lazy() if right_lazy else right)
    assert isinstance(result, pl.LazyFrame if left_lazy or right_lazy else pl.DataFrame)
    if isinstance(result, pl.LazyFrame):
        assert result.collect_schema()["label"] == pl.List(pl.List(pl.Int8))
        result = result.collect()
    expected = left.with_columns(
        pl.Series(
            "label",
            [[[3, 1, 3], [], [None]], [[8]], None, [[None], None, []], []],
            dtype=pl.List(pl.List(pl.Int8)),
        )
    )
    assert_frame_equal(result, expected)


def test_four_levels_with_ancestor_keys_and_block_identity(backend):
    left = pl.DataFrame(
        {
            "conv": [["c1", "c2"]],
            "msg": [[["m1", "m2"], ["m1"]]],
            "block": [[[[0, 1], [0]], [[0]]]],
            "sku": [[[[[7, 7], [7]], [[]]], [[[7, 99]]]]],
            "positions": [[[[[0, 1], [0]], [[]]], [[[0, 1]]]]],
        }
    )
    right = pl.DataFrame(
        {
            "c": ["c1", "c1", "c2"],
            "m": ["m1"] * 3,
            "b": [0, 1, 0],
            "id": [7] * 3,
            "grade": [5, 8, 3],
        },
        schema_overrides={"grade": pl.Int8},
    )
    result = nested_join(
        left,
        right,
        on={"conv": "c", "msg": "m", "block": "b", "sku": "id"},
        shape="positions",
        values={"grade": "label"},
    )
    assert result["label"].to_list() == [[[[[5, 5], [8]], [[]]], [[[3, None]]]]]
    assert_frame_equal(result.select(left.columns), left)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("empty_left", [False, True])
@pytest.mark.parametrize("invalid", ["duplicate", "null"])
def test_all_right_records_validated_even_if_unused(backend, lazy, empty_left, invalid):
    left, right = frames()
    if empty_left:
        left = left.head(0)
    bad = pl.DataFrame(
        {"q": ["unused", "unused"], "id": [999, 999], "grade": [5, 8]}, schema=right.schema
    )
    if invalid == "null":
        bad = bad.head(1).with_columns(pl.lit(None, dtype=pl.Int64).alias("id"))
    right = pl.concat([right, bad])
    with pytest.raises(pl.exceptions.PolarsError, match=f"{invalid}.*right key"):
        if lazy:
            # Projection/filter pushdown must not discard full-right validation.
            join(left.lazy(), right.lazy()).select("unchanged").filter(
                pl.col("unchanged") < 0
            ).collect()
        else:
            join(left, right)


@pytest.mark.parametrize("empty_left,empty_right", [(True, False), (False, True), (True, True)])
def test_empty_inputs_keep_schema(backend, empty_left, empty_right):
    left, right = frames()
    if empty_left:
        left = left.head(0)
    if empty_right:
        right = right.head(0)
    eager = join(left, right)
    lazy = join(left.lazy(), right.lazy()).collect()
    assert_frame_equal(eager, lazy)
    assert eager.schema["label"] == pl.List(pl.List(pl.Int8))
    if empty_right and not empty_left:
        assert eager["label"].to_list() == [
            [[None, None, None], [], [None]],
            [[None]],
            None,
            [[None], None, []],
            [],
        ]


@pytest.mark.parametrize("side", ["left_leaf", "left_parent", "right"])
def test_null_keys_raise_or_skip_without_removing_left_positions(backend, side):
    left, right = frames()
    if side == "left_leaf":
        left = left.head(1).with_columns(
            pl.Series("sku", [[[30, None, 10]]], dtype=left.schema["sku"])
        )
        expected = [[[3, None, 1]]]
    elif side == "left_parent":
        left = left.head(1).with_columns(pl.lit(None, dtype=pl.String).alias("query"))
        expected = [[[None, None, None], [], [None]]]
    else:
        # Duplicated null-key records are all ignored with the opt-in flag.
        bad = pl.DataFrame(
            {"q": [None, None], "id": [30, 30], "grade": [5, 8]}, schema=right.schema
        )
        right = pl.concat([right, bad])
        expected = [[[3, 1, 3], [], [None]], [[8]], None, [[None], None, []], []]
    with pytest.raises(pl.exceptions.PolarsError, match="null in .* key"):
        join(left, right)
    result = join(left, right, skip_null_keys=True)
    assert result["label"].to_list() == expected
    assert_frame_equal(result.select(left.columns), left)


def test_null_ancestor_without_leaf_positions_is_not_a_key(backend):
    left = pl.DataFrame(
        {
            "query": pl.Series([None, None], dtype=pl.String),
            "sku": pl.Series([None, []], dtype=pl.List(pl.List(pl.Int64))),
        }
    )
    right = frames()[1]
    assert join(left, right)["label"].to_list() == [None, []]


@pytest.mark.parametrize(
    "dtype,ids",
    [
        (pl.String, ["x", "y"]),
        (pl.Binary, [b"x", b"y"]),
        (pl.Boolean, [True, False]),
        (pl.Int8, [-1, 2]),
        (pl.Int16, [-300, 300]),
        (pl.UInt8, [1, 255]),
        (pl.UInt16, [1, 65535]),
        (pl.UInt32, [1, 2**32 - 1]),
        (pl.UInt64, [1, 2**64 - 1]),
        (pl.Int128, [-(2**100), 2**100]),
    ],
)
def test_supported_keys_preserve_exact_identity(backend, dtype, ids):
    left = pl.DataFrame({"ids": pl.Series([[ids[1], ids[0], ids[1]]], dtype=pl.List(dtype))})
    right = pl.DataFrame({"id": pl.Series(ids, dtype=dtype), "v": [1, 2]})
    result = nested_join(left, right, on={"ids": "id"}, shape="ids", values={"v": "out"})
    assert result["out"].to_list() == [[2, 1, 2]]


def test_multiple_values_with_nested_and_null_payloads(backend):
    left = pl.DataFrame({"id": [[[2, 1, 99]]]})
    right = pl.DataFrame(
        {
            "id": [1, 2],
            "price": [1.5, None],
            "tags": [["a", None], []],
            "obj": [{"x": 1}, None],
            "nothing": [None, None],
        }
    )
    result = nested_join(
        left,
        right,
        on={"id": "id"},
        shape="id",
        values={"price": "p", "tags": "t", "obj": "o", "nothing": "n"},
    )
    assert result["p"].to_list() == [[[None, 1.5, None]]]
    assert result["t"].to_list() == [[[[], ["a", None], None]]]
    assert result["o"].to_list() == [[[None, {"x": 1}, None]]]
    for source, target in {"price": "p", "tags": "t", "obj": "o", "nothing": "n"}.items():
        assert result.schema[target] == pl.List(pl.List(right.schema[source]))


def test_temporal_payload(backend):
    left = pl.DataFrame({"id": [[1, 2]]})
    right = pl.DataFrame({"id": [1], "date": [date(2025, 1, 1)]})
    result = nested_join(left, right, on={"id": "id"}, shape="id", values={"date": "out"})
    assert result["out"].to_list() == [[date(2025, 1, 1), None]]


@pytest.mark.parametrize(
    "payload",
    [
        pl.Series("v", [[1, 2], None], dtype=pl.Array(pl.Int64, 2)),
        pl.Series("v", [Decimal("1.25"), None], dtype=pl.Decimal(10, 2)),
        pl.Series("v", ["a", None], dtype=pl.Categorical),
        pl.Series("v", ["a", None], dtype=pl.Enum(["a", "b"])),
    ],
)
def test_logical_and_fixed_size_payloads(backend, payload):
    left = pl.DataFrame({"id": [[1, 2, 99]]})
    right = pl.DataFrame({"id": [1, 2], "v": payload})
    result = nested_join(left, right, on={"id": "id"}, shape="id", values={"v": "out"})
    assert result.schema["out"] == pl.List(payload.dtype)
    assert result["out"].to_list() == [[payload.to_list()[0], None, None]]


@pytest.mark.parametrize("key", [[[1]], [[1, 2, 3]], [None]])
def test_structure_mismatch_is_an_error_even_when_skipping_nulls(backend, key):
    left = pl.DataFrame({"shape": [[0, 1]], "key": pl.Series(key, dtype=pl.List(pl.Int64))})
    right = pl.DataFrame({"id": [1], "v": [3]})
    with pytest.raises(pl.exceptions.PolarsError, match="align with shape"):
        nested_join(
            left, right, on={"key": "id"}, shape="shape", values={"v": "out"}, skip_null_keys=True
        )


def test_key_can_stop_at_ancestor_level(backend):
    left = pl.DataFrame({"msg": [[10, 20]], "items": [[[[1, 2]], [[], [3]]]]})
    right = pl.DataFrame({"id": [10, 20], "v": [5, 8]})
    result = nested_join(left, right, on={"msg": "id"}, shape="items", values={"v": "out"})
    assert result["out"].to_list() == [[[[5, 5]], [[], [8]]]]


def test_chunked_and_sliced_inputs(backend):
    left, right = frames()
    chunked = pl.concat([left, left], rechunk=False).slice(1, 7)
    actual = join(chunked, right.reverse())
    expected = pl.concat([join(left, right), join(left, right)]).slice(1, 7)
    assert_frame_equal(actual, expected)


def test_lookup_public_api_shared_and_row_aligned_dictionaries(backend):
    frame = pl.DataFrame({"id": [[1, 2], [1]], "dict": [[{"k": 1, "v": 10}], [{"k": 1, "v": 20}]]})
    expr = nested_lookup("dict", on={"id": "k"}, shape="id", values={"v": "out"})
    assert frame.select(expr).to_series().to_list() == [{"out": [10, None]}, {"out": [20]}]
    dictionary = pl.Series("dict", [[{"k": 1, "v": 30}]], dtype=frame.schema["dict"])
    expr = nested_lookup(pl.lit(dictionary), on={"id": "k"}, shape="id", values={"v": "out"})
    assert frame.lazy().select(expr).collect().to_series().to_list() == [
        {"out": [30, None]},
        {"out": [30]},
    ]


@pytest.mark.parametrize("record", [None, {"k": None, "v": 4}])
def test_null_dictionary_records(backend, record):
    frame = pl.DataFrame(
        {
            "id": [[1]],
            "dict": pl.Series([[record]], dtype=pl.List(pl.Struct({"k": pl.Int64, "v": pl.Int64}))),
        }
    )
    with pytest.raises(pl.exceptions.PolarsError, match="null in right key"):
        frame.select(nested_lookup("dict", on={"id": "k"}, shape="id", values={"v": "out"}))
    out = frame.select(
        nested_lookup("dict", on={"id": "k"}, shape="id", values={"v": "out"}, skip_null_keys=True)
    )
    assert out.to_series().to_list() == [{"out": [None]}]


def test_null_dictionary_is_distinct_from_empty_dictionary(backend):
    frame = pl.DataFrame(
        {
            "id": [[1]],
            "dict": pl.Series([None], dtype=pl.List(pl.Struct({"k": pl.Int64, "v": pl.Int64}))),
        }
    )
    with pytest.raises(pl.exceptions.PolarsError, match="dictionary must not be null"):
        frame.select(
            nested_lookup(
                "dict", on={"id": "k"}, shape="id", values={"v": "out"}, skip_null_keys=True
            )
        )


def test_lazy_construction_does_not_execute_inputs(backend):
    left, right = frames()
    calls = []

    def observe(frame):
        calls.append(frame.height)
        return frame

    result = join(left.lazy().map_batches(observe), right.lazy().map_batches(observe))
    result.collect_schema()
    assert calls == []
    result.collect()
    assert sorted(calls) == sorted([left.height, right.height])


def test_filtered_empty_left_still_validates_entire_right(backend):
    left, right = frames()
    bad = pl.concat([right, right])
    with pytest.raises(pl.exceptions.PolarsError, match="duplicate right key"):
        join(left.lazy().filter(pl.lit(False)), bad.lazy()).collect()


def test_streaming_engine_falls_back_without_losing_dictionary(backend):
    left, right = frames()
    result = join(left.lazy(), right.lazy()).collect(engine="streaming")
    assert_frame_equal(result, join(left, right))


def test_invalid_schema_and_arguments(backend):
    left, right = frames()
    with pytest.raises(pl.exceptions.SchemaError, match="dtype mismatch"):
        join(left, right.with_columns(pl.col("id").cast(pl.Int32)))
    with pytest.raises(pl.exceptions.SchemaError, match="unsupported key dtype"):
        join(
            left.with_columns(pl.col("sku").cast(pl.List(pl.List(pl.Float64)))),
            right.with_columns(pl.col("id").cast(pl.Float64)),
        )
    with pytest.raises(pl.exceptions.DuplicateError, match="already exist"):
        nested_join(left, right, on={"sku": "id"}, shape="sku", values={"grade": "query"})
    with pytest.raises(pl.exceptions.DuplicateError, match="unique"):
        nested_join(left, right, on={"sku": "id"}, shape="sku", values={"grade": "x", "q": "x"})
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        nested_join(left, right, on={"no_column": "id"}, shape="sku", values={"grade": "label"})
    with pytest.raises(TypeError, match="mapping"):
        nested_join(left, right, on="sku", shape="sku", values={"grade": "label"})  # type: ignore
    with pytest.raises(TypeError, match="bool"):
        join(left, right, skip_null_keys=1)


def test_two_independent_sku_and_canonical_joins(backend):
    frame = pl.DataFrame({"sku": [[[[[7], None]]]], "canonical": [[[[None, [7]]]]]})
    sku = pl.DataFrame({"id": [7], "v": [5]})
    canonical = pl.DataFrame({"id": [7], "v": [8]})
    first = nested_join(frame, sku, on={"sku": "id"}, shape="sku", values={"v": "sku_label"})
    result = nested_join(
        first, canonical, on={"canonical": "id"}, shape="canonical", values={"v": "canonical_label"}
    )
    assert result["sku_label"].to_list() == [[[[[5], None]]]]
    assert result["canonical_label"].to_list() == [[[[None, [8]]]]]


@pytest.mark.parametrize("scenario", ["serp", "dialog", "skew"])
def test_explode_reference_matches_nested_join(backend, scenario):
    from performance.native_plugins.nested_join.benchmark import (
        explode_join_implode,
        make_case,
    )

    case = make_case(35, scenario)
    actual = nested_join(case.left, case.right, on=case.on, shape=case.shape, values=case.values)
    assert_frame_equal(actual, explode_join_implode(case))


def test_explode_reference_retains_null_leaf_and_empty_list_distinction(backend):
    from performance.native_plugins.nested_join.benchmark import (
        Case,
        add_indexes,
        explode_join_implode,
    )

    left, right = frames()
    left = left.with_columns(
        pl.Series("sku", [[[None, 10]], [], None, [None, []], [[999]]], dtype=left.schema["sku"])
    )
    left, indexes = add_indexes(left, "sku")
    case = Case(
        left, right, {"query": "q", "sku": "id"}, "sku", {"grade": "label"}, indexes, 3, True
    )
    assert_frame_equal(join(left, right, skip_null_keys=True), explode_join_implode(case))


def test_null_shape_branches_ignore_descendant_key_data(backend):
    from performance.native_plugins.nested_join.benchmark import (
        Case,
        add_indexes,
        explode_join_implode,
    )

    left = pl.DataFrame(
        {
            "shape": pl.Series([None, [None, [0, 0]]], dtype=pl.List(pl.List(pl.Int64))),
            "key": [[[99]], [[99, 99, 99], [2, 1]]],
        }
    )
    right = pl.DataFrame({"id": [1, 2], "v": [5, 8]})
    left, indexes = add_indexes(left, "shape")
    actual = nested_join(left, right, on={"key": "id"}, shape="shape", values={"v": "out"})
    assert actual["out"].to_list() == [None, [None, [8, 5]]]
    case = Case(left, right, {"key": "id"}, "shape", {"v": "out"}, indexes, 2)
    assert_frame_equal(actual, explode_join_implode(case))


def test_direct_lookup_rejects_float_types_and_deeper_keys(backend):
    frame = pl.DataFrame({"shape": [[0]], "key": [[1.0]], "right": [[{"id": 1.0, "v": 5}]]})
    with pytest.raises(pl.exceptions.PolarsError, match="unsupported key dtype"):
        frame.lazy().select(
            nested_lookup("right", on={"key": "id"}, shape="shape", values={"v": "v"})
        ).collect()
    frame = pl.DataFrame({"shape": [[0]], "key": [[[1]]], "right": [[{"id": 1, "v": 5}]]})
    with pytest.raises(pl.exceptions.PolarsError, match="deeper than shape"):
        frame.lazy().select(
            nested_lookup("right", on={"key": "id"}, shape="shape", values={"v": "v"})
        ).collect()


@pytest.mark.parametrize("leaf_first", [False, True])
def test_ancestor_changes_after_skipped_keys_and_between_rows(backend, leaf_first):
    left = pl.DataFrame(
        {
            "group": [["a", None, "b"], ["a", "b", "a"]],
            "id": [
                [[[1, 2], [], [1]], [[1, 2]], [[None, 1], [2]]],
                [[[2]], [[1, 2]], [[1, None, 2]]],
            ],
        }
    )
    right = pl.DataFrame({"g": ["a", "a", "b", "b"], "key": [1, 2, 1, 2], "v": [11, 12, 21, 22]})
    on = {"id": "key", "group": "g"} if leaf_first else {"group": "g", "id": "key"}
    actual = nested_join(left, right, on=on, shape="id", values={"v": "out"}, skip_null_keys=True)
    assert actual["out"].to_list() == [
        [[[11, 12], [], [11]], [[None, None]], [[None, 21], [22]]],
        [[[12]], [[21, 22]], [[11, None, 12]]],
    ]
    assert_frame_equal(actual.select(left.columns), left)
    with pytest.raises(pl.exceptions.PolarsError, match="null in left key"):
        nested_join(left, right, on=on, shape="id", values={"v": "out"})


@pytest.mark.parametrize("empty_right", [False, True])
def test_gather_sliced_complex_payloads_and_missing_matches(backend, empty_right):
    left = pl.DataFrame({"id": [[[3, 1, 3, 99, 2], [], None], None]})
    right = pl.DataFrame(
        {
            "id": [0, 1, 2, 3, 4],
            "text": ["excluded", "long-string-value-1", None, "long-string-value-3", "excluded"],
            "binary": [b"excluded", b"long-binary-value-1\x00", None, b"", b"excluded"],
            "object": [
                {"items": [0], "flag": False},
                {"items": [1, None], "flag": True},
                None,
                {"items": [], "flag": False},
                {"items": [4], "flag": True},
            ],
        }
    ).slice(1, 3)
    if empty_right:
        right = right.head(0)
    actual = nested_join(
        left,
        right,
        on={"id": "id"},
        shape="id",
        values={"text": "text", "binary": "binary", "object": "object"},
    )
    expected_values = {
        "text": ["long-string-value-3", "long-string-value-1", "long-string-value-3", None, None],
        "binary": [b"", b"long-binary-value-1\x00", b"", None, None],
        "object": [
            {"items": [], "flag": False},
            {"items": [1, None], "flag": True},
            {"items": [], "flag": False},
            None,
            None,
        ],
    }
    expected = left.with_columns(
        pl.Series(
            name,
            [[[None] * 5 if empty_right else values, [], None], None],
            dtype=pl.List(pl.List(right.schema[name])),
        )
        for name, values in expected_values.items()
    )
    assert_frame_equal(actual, expected)


def test_sliced_row_dictionaries_do_not_share_matches(backend):
    frame = pl.DataFrame(
        {
            "id": [[1], [1, 2, 1], [1, 2, 1], [1], [1]],
            "dict": pl.Series(
                [
                    [{"k": 1, "v": [0]}],
                    [{"k": 1, "v": [10]}],
                    [{"k": 1, "v": []}, {"k": 2, "v": [20, None]}],
                    [],
                    [{"k": 1, "v": [99]}],
                ],
                dtype=pl.List(pl.Struct({"k": pl.Int64, "v": pl.List(pl.Int64)})),
            ),
        }
    ).slice(1, 3)
    expr = nested_lookup("dict", on={"id": "k"}, shape="id", values={"v": "out"})
    expected = (
        pl.DataFrame(
            {
                "out": pl.Series(
                    [[[10], None, [10]], [[], [20, None], []], [None]],
                    dtype=pl.List(pl.List(pl.Int64)),
                )
            }
        )
        .to_struct("nested_lookup")
        .to_frame()
    )
    assert_frame_equal(frame.select(expr), expected)

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import product

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_list_math import apply_schema, clean_schema, dfs_to_lazy_df, find_null_schema


def test_clean_schema_prunes_empty_parents_preserving_order_and_types() -> None:
    schema = pl.Schema(
        {
            "id": pl.UInt32,
            "drop": pl.List(pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))),
            "keep": pl.Struct(
                {"z": pl.Date, "remove": pl.List(pl.Struct({"a": pl.Int64})), "a": pl.Float32}
            ),
        }
    )
    mask = pl.Schema(
        {
            "drop": pl.List(pl.List(pl.Struct({"a": pl.Null, "b": pl.Null}))),
            "keep": pl.Struct({"remove": pl.List(pl.Struct({"a": pl.Null}))}),
        }
    )
    original_schema, original_mask = pl.Schema(schema), pl.Schema(mask)

    assert clean_schema(schema, mask) == pl.Schema(
        {"id": pl.UInt32, "keep": pl.Struct({"z": pl.Date, "a": pl.Float32})}
    )
    assert schema == original_schema
    assert mask == original_mask
    assert clean_schema(schema, pl.Schema()) == schema


def test_null_marker_removes_a_whole_subtree() -> None:
    schema = pl.Schema({"x": pl.List(pl.Struct({"a": pl.Int64})), "y": pl.String})
    assert clean_schema(schema, pl.Schema({"x": pl.Null})) == pl.Schema({"y": pl.String})


@pytest.mark.parametrize(
    "schema,mask,match",
    [
        ({"x": pl.Int64}, {"unknown": pl.Null}, "Unknown names"),
        ({"x": pl.Struct({"a": pl.Int64})}, {"x": pl.Struct({"b": pl.Null})}, "Unknown names"),
        ({"x": pl.Int64}, {"x": pl.String}, "Invalid subschema"),
        ({"x": pl.Int64}, {"x": pl.List(pl.Null)}, "Invalid subschema"),
        ({"x": pl.List(pl.Int64)}, {"x": pl.Struct({"a": pl.Null})}, "Invalid subschema"),
        ({"x": pl.Array(pl.Int64, 2)}, {"x": pl.Array(pl.Null, 3)}, "Invalid subschema"),
        ({"x": pl.List(pl.Int64)}, {"x": pl.List(pl.String)}, "Invalid subschema"),
        ({"x": pl.Datetime("us")}, {"x": pl.Datetime("ns")}, "Invalid subschema"),
    ],
)
def test_rejects_invalid_masks(schema: dict, mask: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        clean_schema(pl.Schema(schema), pl.Schema(mask))


def test_typed_subschema_removes_populated_fields() -> None:
    frame = pl.LazyFrame(
        {"id": [1], "items": [[{"keep": "value", "remove": 42}]]},
        schema={
            "id": pl.Int64,
            "items": pl.List(pl.Struct({"keep": pl.String, "remove": pl.Int64})),
        },
    )
    subschema = pl.Schema({"id": pl.Int64, "items": pl.List(pl.Struct({"remove": pl.Int64}))})
    original = pl.Schema(subschema)
    target = clean_schema(frame.collect_schema(), subschema=subschema)

    assert target == pl.Schema({"items": pl.List(pl.Struct({"keep": pl.String}))})
    assert subschema == original
    assert apply_schema(frame, target).collect().to_dict(as_series=False) == {
        "items": [[{"keep": "value"}]]
    }


def test_typed_and_null_leaves_can_be_combined() -> None:
    schema = pl.Schema({"x": pl.Struct({"keep": pl.UInt32, "a": pl.Int64, "b": pl.String})})
    subschema = pl.Schema({"x": pl.Struct({"a": pl.Int64, "b": pl.Null})})
    assert clean_schema(schema, subschema) == pl.Schema({"x": pl.Struct({"keep": pl.UInt32})})


def test_typed_subschema_prunes_nested_containers() -> None:
    dtype = pl.Struct({"a": pl.Int64, "b": pl.String})
    schema = pl.Schema(
        {"keep": pl.UInt32, "lists": pl.List(pl.List(dtype)), "arrays": pl.Array(dtype, (2, 3))}
    )
    subschema = pl.Schema({"arrays": pl.Array(dtype, (2, 3)), "lists": pl.List(pl.List(dtype))})
    assert clean_schema(schema, subschema) == pl.Schema({"keep": pl.UInt32})
    assert clean_schema(schema, schema) == pl.Schema()


def test_round_trip_preserves_nested_nulls_and_list_lengths() -> None:
    dtype = pl.List(pl.List(pl.Struct({"a": pl.Int64, "b": pl.String})))
    frame = pl.LazyFrame(
        {
            "x": [None, [], [None, [], [None, {"a": 1, "b": None}, {"a": None, "b": None}]]],
            "drop": [None] * 3,
        },
        schema={"x": dtype, "drop": pl.String},
    )
    target = clean_schema(frame.collect_schema(), find_null_schema(frame))
    result = apply_schema(frame, target)

    assert isinstance(result, pl.LazyFrame)
    assert result.collect_schema() == target
    assert_frame_equal(
        result.collect(),
        pl.DataFrame(
            {"x": [None, [], [None, [], [None, {"a": 1}, {"a": None}]]]},
            schema={"x": pl.List(pl.List(pl.Struct({"a": pl.Int64})))},
        ),
    )


def test_struct_list_struct_round_trip() -> None:
    frame = pl.LazyFrame(
        {"x": [None, {"items": [None, {"keep": 1, "drop": None}], "drop": None}]},
        schema={
            "x": pl.Struct(
                {"items": pl.List(pl.Struct({"keep": pl.Int64, "drop": pl.Date})), "drop": pl.Int64}
            )
        },
    )
    target = clean_schema(frame.collect_schema(), find_null_schema(frame))
    assert apply_schema(frame, target).collect().to_dict(as_series=False) == {
        "x": [None, {"items": [None, {"keep": 1}]}]
    }


def test_array_round_trip() -> None:
    frame = pl.LazyFrame(
        {"x": [None, [[None, {"a": 1, "b": None}], [None, None]]]},
        schema={"x": pl.Array(pl.Struct({"a": pl.Int64, "b": pl.String}), (2, 2))},
    )
    target = clean_schema(frame.collect_schema(), find_null_schema(frame))
    assert target == pl.Schema({"x": pl.Array(pl.Struct({"a": pl.Int64}), (2, 2))})
    result = apply_schema(frame, target).collect()
    assert result.schema == target
    assert result.to_dict(as_series=False) == {"x": [None, [[None, {"a": 1}], [None, None]]]}


def test_apply_schema_is_lazy_and_reorders_and_casts() -> None:
    scans: list[int] = []
    schema = pl.Schema({"a": pl.Int64, "b": pl.Int64})

    def batches() -> Iterator[pl.DataFrame]:
        scans.append(1)
        yield pl.DataFrame({"a": [1, 2], "b": [3, 4]}, schema=schema)

    frame = dfs_to_lazy_df(batches(), schema)
    target = pl.Schema({"b": pl.Float64, "a": pl.Int32})
    result = apply_schema(frame, target)
    assert result.collect_schema() == target
    assert not scans
    assert_frame_equal(
        result.collect(), pl.DataFrame({"b": [3.0, 4.0], "a": [1, 2]}, schema=target)
    )
    assert len(scans) == 1


def test_apply_schema_uses_strict_casts() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        apply_schema(pl.LazyFrame({"x": ["invalid"]}), pl.Schema({"x": pl.Int64})).collect()


def test_empty_result_and_empty_input() -> None:
    for frame in [
        pl.LazyFrame(),
        pl.LazyFrame({"x": [None]}),
        pl.LazyFrame(schema={"x": pl.Int64}),
    ]:
        target = clean_schema(frame.collect_schema(), find_null_schema(frame))
        assert target == pl.Schema()
        assert apply_schema(frame, target).collect().shape == (0, 0)
    frame = pl.LazyFrame(schema={"x": pl.Struct({"a": pl.Int64, "b": pl.String})})
    target = pl.Schema({"x": pl.Struct({"a": pl.Int64})})
    assert apply_schema(frame, target).collect().schema == target


def test_apply_schema_uses_literal_names() -> None:
    frame = pl.LazyFrame({"*": [1], "^x$": [2], "x": [3]})
    result = apply_schema(frame, pl.Schema({"^x$": pl.Int64, "*": pl.Int64})).collect()
    assert result.to_dict(as_series=False) == {"^x$": [2], "*": [1]}


@pytest.mark.parametrize("containers", list(product(["list", "array", "struct"], repeat=3)))
def test_mixed_containers_round_trip(containers: tuple[str, ...]) -> None:
    dtype: pl.DataType = pl.Struct({"keep": pl.Int64, "drop": pl.String})
    target_dtype: pl.DataType = pl.Struct({"keep": pl.Int64})
    value: object = {"keep": 7, "drop": None}
    expected: object = {"keep": 7}
    for container in containers:
        if container == "struct":
            dtype, target_dtype = pl.Struct({"node": dtype}), pl.Struct({"node": target_dtype})
            value, expected = {"node": value}, {"node": expected}
        else:
            if container == "list":
                dtype, target_dtype = pl.List(dtype), pl.List(target_dtype)
            else:
                dtype, target_dtype = pl.Array(dtype, 2), pl.Array(target_dtype, 2)
            value, expected = [None, value], [None, expected]

    frame = pl.LazyFrame({"x": [None, value]}, schema={"x": dtype})
    target = clean_schema(frame.collect_schema(), find_null_schema(frame))
    assert target == pl.Schema({"x": target_dtype})
    result = apply_schema(frame, target).collect()
    assert_frame_equal(result, pl.DataFrame({"x": [None, expected]}, schema=target))
    # Discovery and application should stabilize after absent fields are removed.
    assert find_null_schema(result.lazy()) == pl.Schema()
    assert_frame_equal(apply_schema(result.lazy(), target).collect(), result)


def test_no_absent_fields_leaves_frame_unchanged() -> None:
    frame = pl.DataFrame(
        {"id": [None, 1], "items": [[], [{"a": 1, "b": "value"}]]},
        schema={"id": pl.Int64, "items": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String}))},
    )
    mask = find_null_schema(frame.lazy())
    assert mask == pl.Schema()
    target = clean_schema(frame.schema, mask)
    assert target == frame.schema
    assert_frame_equal(apply_schema(frame.lazy(), target).collect(), frame)


def test_empty_structs_are_pruned_through_parent_containers() -> None:
    frame = pl.LazyFrame(
        {"id": [1], "drop": [[{}]], "keep": [{"a": 2, "empty": {}}]},
        schema={
            "id": pl.Int64,
            "drop": pl.List(pl.Struct({})),
            "keep": pl.Struct({"a": pl.Int64, "empty": pl.Struct({})}),
        },
    )
    target = clean_schema(frame.collect_schema(), find_null_schema(frame))
    assert target == pl.Schema({"id": pl.Int64, "keep": pl.Struct({"a": pl.Int64})})
    assert apply_schema(frame, target).collect().to_dict(as_series=False) == {
        "id": [1],
        "keep": [{"a": 2}],
    }


def test_subschema_field_order_does_not_change_retained_order() -> None:
    schema = pl.Schema(
        {"x": pl.Struct({"a": pl.Int64, "b": pl.String, "c": pl.Boolean, "d": pl.UInt32})}
    )
    subschema = pl.Schema({"x": pl.Struct({"d": pl.UInt32, "b": pl.String})})
    assert clean_schema(schema, subschema) == pl.Schema(
        {"x": pl.Struct({"a": pl.Int64, "c": pl.Boolean})}
    )
    assert clean_schema(schema, pl.Schema({"x": pl.Struct({})})) == schema


@pytest.mark.parametrize(
    "dtype,value",
    [
        (pl.Datetime("us", "UTC"), datetime(2026, 1, 1, tzinfo=timezone.utc)),
        (pl.Duration("ms"), timedelta(seconds=1)),
        (pl.Decimal(10, 2), Decimal("1.25")),
        (pl.Enum(["a", "b"]), "a"),
        (pl.Categorical(), "a"),
        (pl.Binary(), b"value"),
    ],
)
def test_leaf_metadata_survives_cleaning(dtype: pl.DataType, value: object) -> None:
    frame = pl.LazyFrame(
        {"x": [[{"keep": value, "drop": None}]]},
        schema={"x": pl.List(pl.Struct({"keep": dtype, "drop": dtype}))},
    )
    mask = find_null_schema(frame)
    assert mask == pl.Schema({"x": pl.List(pl.Struct({"drop": pl.Null}))})
    target = clean_schema(frame.collect_schema(), mask)
    assert target == pl.Schema({"x": pl.List(pl.Struct({"keep": dtype}))})
    assert_frame_equal(
        apply_schema(frame, target).collect(),
        pl.DataFrame({"x": [[{"keep": value}]]}, schema=target),
    )


def test_unknown_field_and_array_shape_errors_inside_nested_subschemas() -> None:
    schema = pl.Schema({"x": pl.List(pl.Struct({"a": pl.Array(pl.Int64, (2, 3))}))})
    with pytest.raises(ValueError, match="Unknown names.*missing"):
        clean_schema(schema, pl.Schema({"x": pl.List(pl.Struct({"missing": pl.Null}))}))
    with pytest.raises(ValueError, match="Invalid subschema"):
        clean_schema(
            schema, pl.Schema({"x": pl.List(pl.Struct({"a": pl.Array(pl.Int64, (2, 4))}))})
        )


def test_apply_schema_missing_columns_and_nested_fields() -> None:
    frame = pl.LazyFrame({"x": [None, {"a": 1}]}, schema={"x": pl.Struct({"a": pl.Int64})})
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        apply_schema(frame, pl.Schema({"missing": pl.String})).collect()
    target = pl.Schema({"x": pl.Struct({"a": pl.Int64, "b": pl.String})})
    assert_frame_equal(
        apply_schema(frame, target).collect(),
        pl.DataFrame({"x": [None, {"a": 1, "b": None}]}, schema=target),
    )

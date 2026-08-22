from dataclasses import dataclass, field
from typing import assert_type

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


class DynamicSchema(tp.Schema):
    row_id = tp.Column[int](polars_name="rowId")
    metrics = tp.FlatDict[float]()


@tp.model(schema=DynamicSchema)
class DynamicRow:
    row_id: int
    metrics: dict[str, float] = field(default_factory=dict)


def test_flat_dict_uses_context_keys_as_physical_columns() -> None:
    context = tp.Context().bind(DynamicSchema.metrics, ["views", "conversion"])
    rows = [
        DynamicRow(1, {"views": 10.0}),
        DynamicRow(2, {"conversion": 0.25}),
    ]

    frame = DynamicSchema.to_frame_many(DynamicRow, rows, context=context)

    assert frame.schema == pl.Schema(
        {
            "rowId": pl.Int64,
            "metrics_views": pl.Float64,
            "metrics_conversion": pl.Float64,
        }
    )
    assert frame.to_dicts() == [
        {"rowId": 1, "metrics_views": 10.0, "metrics_conversion": None},
        {"rowId": 2, "metrics_views": None, "metrics_conversion": 0.25},
    ]
    assert list(DynamicSchema.iter_frame(DynamicRow, frame)) == rows
    assert list(DynamicSchema.iter_frame(DynamicRow, frame, strict_schema=True)) == rows
    assert DynamicSchema.polars_schema(context=context) == frame.schema
    assert DynamicSchema.model_schema(DynamicRow, context=context) == frame.schema
    assert_type(DynamicSchema.metrics, tp.FlatDict[float])


def test_absent_context_means_no_flat_dict_columns() -> None:
    row = DynamicRow(1)

    frame = DynamicSchema.to_frame(row)

    assert frame.schema == pl.Schema({"rowId": pl.Int64})
    assert DynamicSchema.polars_schema() == frame.schema
    assert DynamicSchema.from_frame(DynamicRow, frame, strict_schema=True) == row


def test_flat_dict_rejects_keys_not_bound_in_context() -> None:
    context = tp.Context().bind(DynamicSchema.metrics, ["known"])

    with pytest.raises(TypeError, match=r"unbound key.*'unknown'"):
        DynamicSchema.to_frame(
            DynamicRow(1, {"known": 1.0, "unknown": 2.0}),
            context=context,
        )
    with pytest.raises(TypeError, match=r"unbound key.*'known'"):
        DynamicSchema.to_frame(DynamicRow(1, {"known": 1.0}))


def test_flat_dict_physical_dict_conversion_uses_context_only_forward() -> None:
    context = tp.Context().bind(DynamicSchema.metrics, ["first", "second"])
    row = DynamicRow(1, {"first": 0.5})

    physical = DynamicSchema.to_dict(
        row,
        by_polars_name=True,
        context=context,
    )

    assert DynamicSchema.to_dict(row) == {
        "row_id": 1,
        "metrics": {"first": 0.5},
    }
    assert physical == {
        "rowId": 1,
        "metrics_first": 0.5,
        "metrics_second": None,
    }
    assert (
        DynamicSchema.from_dict(
            DynamicRow,
            physical,
            by_polars_name=True,
        )
        == row
    )


def test_flat_dict_supports_nested_and_flat_structs() -> None:
    class DetailsSchema(tp.Schema):
        label = tp.Column[str]()
        values = tp.FlatDict[int](divider="__")

    class NestedSchema(tp.Schema):
        details = tp.Struct[DetailsSchema]()

    class FlatSchema(tp.Schema):
        details = tp.FlatStruct[DetailsSchema]()

    class FlatListSchema(tp.Schema):
        details = tp.FlatListStruct[DetailsSchema]()

    @dataclass
    class Details:
        label: str
        values: dict[str, int] = field(default_factory=dict)

    @tp.model(schema=NestedSchema)
    class NestedRow:
        details: Details

    @tp.model(schema=FlatSchema)
    class FlatRow:
        details: Details

    @tp.model(schema=FlatListSchema)
    class FlatListRow:
        details: list[Details]

    context = tp.Context().bind(DetailsSchema.values, ["left", "right"])
    nested_row = NestedRow(Details("nested", {"left": 1}))
    flat_row = FlatRow(Details("flat", {"right": 2}))
    flat_list_row = FlatListRow([Details("first", {"left": 1}), Details("second", {"right": 2})])

    nested_frame = NestedSchema.to_frame(nested_row, context=context)
    flat_frame = FlatSchema.to_frame(flat_row, context=context)
    flat_list_frame = FlatListSchema.to_frame(flat_list_row, context=context)

    assert nested_frame.schema == pl.Schema(
        {
            "details": pl.Struct(
                {
                    "label": pl.String,
                    "values__left": pl.Int64,
                    "values__right": pl.Int64,
                }
            )
        }
    )
    assert flat_frame.columns == [
        "details_label",
        "details_values__left",
        "details_values__right",
    ]
    assert NestedSchema.from_frame(NestedRow, nested_frame, strict_schema=True) == nested_row
    assert FlatSchema.from_frame(FlatRow, flat_frame, strict_schema=True) == flat_row
    assert (
        FlatListSchema.from_frame(
            FlatListRow,
            flat_list_frame,
            strict_schema=True,
        )
        == flat_list_row
    )
    assert (
        FlatSchema.details.fields.values.key_expr("right").meta.output_name()
        == "details_values__right"
    )


def test_context_and_flat_dict_declarations_are_validated() -> None:
    assert isinstance(DynamicSchema.metrics, tp.ContextFieldProtocol)
    assert not isinstance(DynamicSchema.row_id, tp.ContextFieldProtocol)
    with pytest.raises(TypeError, match="ContextFieldProtocol"):
        tp.Context().bind(DynamicSchema.row_id, ["value"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="iterable of strings"):
        tp.Context().bind(DynamicSchema.metrics, "value")
    with pytest.raises(TypeError, match="non-empty strings"):
        tp.Context().bind(DynamicSchema.metrics, [""])
    with pytest.raises(TypeError, match="unique"):
        tp.Context().bind(DynamicSchema.metrics, ["same", "same"])
    with pytest.raises(TypeError, match="divider"):
        tp.FlatDict[int](divider="")

    class WrongTypeSchema(tp.Schema):
        values = tp.FlatDict[float]()

    with pytest.raises(TypeError, match="uses Python type"):

        @tp.model(schema=WrongTypeSchema)
        class WrongType:
            values: dict[str, int]


def test_flat_dict_detects_runtime_physical_name_conflicts() -> None:
    class ConflictingSchema(tp.Schema):
        metrics = tp.FlatDict[int]()
        existing = tp.Column[int](polars_name="metrics_value")

    context = tp.Context().bind(ConflictingSchema.metrics, ["value"])

    with pytest.raises(TypeError, match="conflicting physical field"):
        ConflictingSchema.polars_schema(context=context)


def test_builder_caches_flat_dict_plan_by_context_snapshot() -> None:
    builder = tp.Builder.for_model(DynamicRow, schema=DynamicSchema)
    context = tp.Context().bind(DynamicSchema.metrics, ["first"])

    first = builder.physical_plan_for(context)
    second = builder.physical_plan_for(context)
    context.bind(DynamicSchema.metrics, ["second"])
    changed = builder.physical_plan_for(context)

    assert first is second
    assert changed is not first
    assert first.schema.names() == ["rowId", "metrics_first"]
    assert changed.schema.names() == ["rowId", "metrics_second"]


def test_context_can_bind_reused_nested_schema_paths_independently() -> None:
    class ValuesSchema(tp.Schema):
        values = tp.FlatDict[int]()

    class PairSchema(tp.Schema):
        left = tp.Struct[ValuesSchema]()
        right = tp.Struct[ValuesSchema]()

    @dataclass
    class Values:
        values: dict[str, int] = field(default_factory=dict)

    @tp.model(schema=PairSchema)
    class Pair:
        left: Values
        right: Values

    context = (
        tp.Context()
        .bind(PairSchema.left.fields.values, ["left_key"])
        .bind(PairSchema.right.fields.values, ["right_key"])
    )
    row = Pair(Values({"left_key": 1}), Values({"right_key": 2}))

    frame = PairSchema.to_frame(row, context=context)

    assert frame.schema == pl.Schema(
        {
            "left": pl.Struct({"values_left_key": pl.Int64}),
            "right": pl.Struct({"values_right_key": pl.Int64}),
        }
    )
    assert PairSchema.polars_schema(context=context) == frame.schema
    assert PairSchema.from_frame(Pair, frame, strict_schema=True) == row

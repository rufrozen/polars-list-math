from dataclasses import dataclass
from typing import assert_type

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


class DynamicSchema(tp.Schema):
    row_id = tp.Column[int](polars_name="rowId")
    scores = tp.FlatTuple[tp.F32](divider="__")


@tp.model(schema=DynamicSchema)
class DynamicRow:
    row_id: int
    scores: tuple[float, ...] = ()


def test_flat_tuple_uses_context_names_as_ordered_physical_columns() -> None:
    context = tp.Context().bind(DynamicSchema.scores, ["precision", "recall"])
    row = DynamicRow(1, (0.75, 0.5))

    frame = DynamicSchema.to_frame(row, context=context)

    assert frame.schema == pl.Schema(
        {
            "rowId": pl.Int64,
            "scores__precision": pl.Float32,
            "scores__recall": pl.Float32,
        }
    )
    assert frame.row(0) == (1, 0.75, 0.5)
    assert DynamicSchema.from_frame(DynamicRow, frame, strict_schema=True) == row
    assert DynamicSchema.polars_schema(context=context) == frame.schema
    assert DynamicSchema.model_schema(DynamicRow, context=context) == frame.schema
    assert DynamicSchema.scores.key_expr("recall").meta.output_name() == "scores__recall"
    assert_type(DynamicSchema.scores, tp.FlatTuple[tp.F32])


def test_flat_tuple_physical_dict_conversion_preserves_position_order() -> None:
    context = tp.Context().bind(DynamicSchema.scores, ["first", "second"])
    row = DynamicRow(1, (0.25, 0.75))

    physical = DynamicSchema.to_dict(row, by_polars_name=True, context=context)

    assert DynamicSchema.to_dict(row) == {
        "row_id": 1,
        "scores": (0.25, 0.75),
    }
    assert physical == {
        "rowId": 1,
        "scores__first": 0.25,
        "scores__second": 0.75,
    }
    assert DynamicSchema.from_dict(DynamicRow, physical, by_polars_name=True) == row


def test_absent_context_means_an_empty_flat_tuple() -> None:
    row = DynamicRow(1)

    frame = DynamicSchema.to_frame(row)

    assert frame.schema == pl.Schema({"rowId": pl.Int64})
    assert DynamicSchema.from_frame(DynamicRow, frame, strict_schema=True) == row


def test_flat_tuple_requires_a_tuple_matching_the_context_length() -> None:
    context = tp.Context().bind(DynamicSchema.scores, ["first", "second"])

    with pytest.raises(TypeError, match=r"has 1 value.*binds 2 position"):
        DynamicSchema.to_frame(DynamicRow(1, (0.5,)), context=context)
    with pytest.raises(TypeError, match=r"has 1 value.*binds 0 position"):
        DynamicSchema.to_frame(DynamicRow(1, (0.5,)))

    row = DynamicRow(1)
    row.scores = [0.25, 0.75]  # type: ignore[assignment]
    with pytest.raises(TypeError, match="must be a tuple"):
        DynamicSchema.to_frame(row, context=context)


def test_flat_tuple_supports_nested_flat_struct_paths() -> None:
    class ScoresSchema(tp.Schema):
        label = tp.Column[str]()
        values = tp.FlatTuple[int]()

    class PairSchema(tp.Schema):
        left = tp.FlatStruct[ScoresSchema](divider="__")
        right = tp.Struct[ScoresSchema]()

    @dataclass
    class Scores:
        label: str
        values: tuple[int, ...] = ()

    @tp.model(schema=PairSchema)
    class Pair:
        left: Scores
        right: Scores

    context = (
        tp.Context()
        .bind(PairSchema.left.fields.values, ["low", "high"])
        .bind(PairSchema.right.fields.values, ["only"])
    )
    row = Pair(Scores("left", (1, 2)), Scores("right", (3,)))

    frame = PairSchema.to_frame(row, context=context)

    assert frame.schema == pl.Schema(
        {
            "left__label": pl.String,
            "left__values_low": pl.Int64,
            "left__values_high": pl.Int64,
            "right": pl.Struct(
                {
                    "label": pl.String,
                    "values_only": pl.Int64,
                }
            ),
        }
    )
    assert PairSchema.from_frame(Pair, frame, strict_schema=True) == row
    assert PairSchema.left.fields.values.key_expr("high").meta.output_name() == (
        "left__values_high"
    )


def test_flat_tuple_declaration_requires_matching_variable_tuple_type() -> None:
    class WrongTypeSchema(tp.Schema):
        values = tp.FlatTuple[float]()

    with pytest.raises(TypeError, match="uses Python type"):

        @tp.model(schema=WrongTypeSchema)
        class WrongType:
            values: tuple[int, ...]

    with pytest.raises(TypeError, match="uses Python type"):

        @tp.model(schema=WrongTypeSchema)
        class FixedTuple:
            values: tuple[float, float]

    with pytest.raises(TypeError, match="divider"):
        tp.FlatTuple[int](divider="")

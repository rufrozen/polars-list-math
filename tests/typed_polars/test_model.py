from dataclasses import dataclass, field

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


class SuggestionSchema(tp.Schema):
    value = tp.Column[str]()
    corrected_query = tp.Column[str](polars_name="correctedQuery")


class CompletionSchema(tp.Schema):
    prefix = tp.Column[str](polars_name="queryPrefix")
    suggestions = tp.ListStruct[SuggestionSchema]()


class RowSchema(tp.Schema):
    request_id = tp.Column[str](polars_name="requestId")
    position = tp.Column[int](dtype=pl.Int32)
    completion = tp.Struct[CompletionSchema]()
    tags = tp.Column[list[str]]()


@tp.model(schema=SuggestionSchema)
class Suggestion:
    value: str
    corrected_query: str


@tp.model(schema=CompletionSchema)
class Completion:
    prefix: str
    suggestions: list[Suggestion] = field(default_factory=list)


@tp.model(schema=RowSchema)
class Row:
    request_id: str
    position: int
    completion: Completion
    tags: list[str] = field(default_factory=list)


def make_row(request_id: str = "one") -> Row:
    return Row(
        request_id,
        7,
        Completion("по", [Suggestion("поле", "поле"), Suggestion("полёт", "полет")]),
    )


def test_schema_supplies_all_polars_names_and_dtypes() -> None:
    frame = RowSchema.to_frame(make_row())
    assert frame.schema == pl.Schema(
        {
            "requestId": pl.String,
            "position": pl.Int32,
            "completion": pl.Struct(
                {
                    "queryPrefix": pl.String,
                    "suggestions": pl.List(
                        pl.Struct({"value": pl.String, "correctedQuery": pl.String})
                    ),
                }
            ),
            "tags": pl.List(pl.String),
        }
    )
    assert RowSchema.from_frame(Row, frame) == make_row()


def test_many_rows_and_dict_round_trip() -> None:
    rows = [make_row("one"), make_row("two")]
    frame = RowSchema.to_frame_many(Row, rows)
    assert list(RowSchema.iter_frame(Row, frame)) == rows
    physical = RowSchema.to_dict(rows[0], by_polars_name=True)
    assert physical["completion"]["queryPrefix"] == "по"
    assert RowSchema.from_dict(Row, physical, by_polars_name=True) == rows[0]


def test_standard_slots_dataclass_is_supported() -> None:
    class FrozenSchema(tp.Schema):
        value = tp.Column[str]()
        corrected_query = tp.Column[str](polars_name="correctedQuery")

    @tp.model(schema=FrozenSchema)
    @dataclass(slots=True, frozen=True)
    class Frozen:
        value: str
        corrected_query: str = "same"

    assert FrozenSchema.to_frame(Frozen("one")).schema == FrozenSchema.polars_schema()


def test_only_root_dataclass_requires_model_decorator() -> None:
    @dataclass
    class PlainSuggestion:
        value: str
        corrected_query: str

    class PlainSchema(tp.Schema):
        prefix = tp.Column[str](polars_name="queryPrefix")
        suggestions = tp.ListStruct[SuggestionSchema]()

    @tp.model(schema=PlainSchema)
    @dataclass
    class PlainCompletion:
        prefix: str
        suggestions: list[PlainSuggestion] = field(default_factory=list)

    row = PlainCompletion("one", [PlainSuggestion("value", "corrected")])
    frame = PlainSchema.to_frame(row)

    assert PlainSchema.from_frame(PlainCompletion, frame) == row


def test_schema_to_frame_many_supports_empty_rows() -> None:
    frame = RowSchema.to_frame_many(Row, [])
    assert frame.schema == tp.Builder.for_model(Row, schema=RowSchema).polars_schema()
    builder = tp.Builder.for_model(Row, schema=RowSchema)
    assert isinstance(builder, tp.Builder)
    assert builder is tp.Builder.for_model(Row, schema=RowSchema)
    assert builder.polars_schema() == frame.schema


def test_model_decorator_does_not_require_model_base_class() -> None:
    class DecoratedSchema(tp.Schema):
        value = tp.Column[str]()
        corrected_query = tp.Column[str](polars_name="correctedQuery")

    @tp.model(schema=DecoratedSchema)
    class Decorated:
        value: str
        corrected_query: str

    row = Decorated("one", "same")
    frame = DecoratedSchema.to_frame(row)

    assert DecoratedSchema.from_frame(Decorated, frame) == row

    @dataclass
    class Wrong:
        value: str
        corrected_query: str

    with pytest.raises(TypeError, match="Wrong has no builder"):
        DecoratedSchema.to_frame(Wrong("one", "same"))
    with pytest.raises(TypeError, match="Wrong has no builder"):
        DecoratedSchema.from_frame(Wrong, frame)

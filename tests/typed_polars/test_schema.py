from dataclasses import dataclass, field
from typing import assert_type

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


class LocationSchema(tp.Schema):
    city = tp.Column[str]()
    country_code = tp.Column[str](polars_name="countryCode")


class ContextSchema(tp.Schema):
    source = tp.Column[str]()
    location = tp.Struct[LocationSchema]()


class EventSchema(tp.Schema):
    name = tp.Column[str]()
    score = tp.Column[tp.F32]()
    context = tp.Struct[ContextSchema]()


class DataFrameSchema(tp.Schema):
    request_id = tp.Column[str](polars_name="requestId")
    context = tp.Struct[ContextSchema]()
    events = tp.ListStruct[EventSchema]()
    tags = tp.Column[list[str]]()
    optional_comment = tp.Column[str | None](polars_name="optionalComment")
    unused = tp.Column[bool]()


def test_schema_builds_complete_logical_schema() -> None:
    assert DataFrameSchema.polars_schema() == pl.Schema(
        {
            "requestId": pl.String,
            "context": pl.Struct(
                {
                    "source": pl.String,
                    "location": pl.Struct({"city": pl.String, "countryCode": pl.String}),
                }
            ),
            "events": pl.List(
                pl.Struct(
                    {
                        "name": pl.String,
                        "score": pl.Float32,
                        "context": pl.Struct(
                            {
                                "source": pl.String,
                                "location": pl.Struct(
                                    {"city": pl.String, "countryCode": pl.String}
                                ),
                            }
                        ),
                    }
                )
            ),
            "tags": pl.List(pl.String),
            "optionalComment": pl.String,
            "unused": pl.Boolean,
        }
    )
    assert DataFrameSchema.fields()["request_id"] is DataFrameSchema.request_id


def test_schema_builds_typed_nested_polars_expressions() -> None:
    frame = pl.DataFrame(
        {
            "requestId": ["one"],
            "context": [
                {
                    "source": "search",
                    "location": {"city": "Moscow", "countryCode": "RU"},
                }
            ],
            "events": [
                [
                    {
                        "name": "open",
                        "score": 0.75,
                        "context": {
                            "source": "web",
                            "location": {"city": "Berlin", "countryCode": "DE"},
                        },
                    }
                ]
            ],
            "tags": [["typed"]],
            "optionalComment": [None],
            "unused": [True],
        },
        schema=DataFrameSchema.polars_schema(),
    )

    result = frame.select(
        DataFrameSchema.request_id.expr(),
        DataFrameSchema.context.fields.location.fields.city.expr().alias("city"),
        DataFrameSchema.events.item.score.expr().alias("eventScores"),
        DataFrameSchema.events.item.context.fields.location.fields.country_code.expr().alias(
            "eventCountryCodes"
        ),
    )

    assert result.to_dicts() == [
        {
            "requestId": "one",
            "city": "Moscow",
            "eventScores": [0.75],
            "eventCountryCodes": ["DE"],
        }
    ]
    assert_type(DataFrameSchema.request_id, tp.Column[str])
    assert_type(DataFrameSchema.context.fields.location.fields.city, tp.Column[str])
    assert_type(DataFrameSchema.events.item.score, tp.Column[tp.F32])


def test_model_can_validate_a_recursive_schema_subset() -> None:
    @dataclass
    class Location:
        city: str

    @dataclass
    class Context:
        source: str
        location: Location

    @dataclass
    class Event:
        name: str
        score: float

    @tp.model(schema=DataFrameSchema)
    class Row:
        request_id: str
        context: Context
        events: list[Event]
        tags: list[str] = field(default_factory=list)
        optional_comment: str | None = None

    model_schema = tp.Builder.for_model(Row, schema=DataFrameSchema).polars_schema()
    assert DataFrameSchema.model_schema(Row) == model_schema
    assert model_schema["requestId"] == pl.String
    assert model_schema["events"] == pl.List(pl.Struct({"name": pl.String, "score": pl.Float32}))
    assert Row("one", Context("web", Location("Moscow")), []).optional_comment is None


def test_flat_struct_and_list_struct_are_declared_by_schema() -> None:
    class FlatSchema(tp.Schema):
        context = tp.Struct[ContextSchema](flat=True, flat_divider="__")
        events = tp.ListStruct[EventSchema](flat=True)

    @dataclass
    class Location:
        city: str
        country_code: str

    @dataclass
    class Context:
        source: str
        location: Location

    @dataclass
    class Event:
        name: str
        score: float
        context: Context

    @tp.model(schema=FlatSchema)
    class Row:
        context: Context
        events: list[Event]

    context = Context("search", Location("Moscow", "RU"))
    row = Row(context, [Event("open", 0.75, context)])
    frame = FlatSchema.to_frame(row)

    assert frame.schema == FlatSchema.polars_schema()
    assert frame.columns == [
        "context__source",
        "context__location",
        "events_name",
        "events_score",
        "events_context",
    ]
    assert FlatSchema.from_frame(Row, frame) == row
    selected = frame.select(
        FlatSchema.context.fields.source.expr(),
        FlatSchema.events.item.score.expr(),
    )
    assert selected.to_dicts() == [{"context__source": "search", "events_score": [0.75]}]


def test_flat_schema_options_are_validated() -> None:
    with pytest.raises(TypeError, match="flat must be a bool"):
        tp.Struct[ContextSchema](flat=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="flat_divider"):
        tp.ListStruct[EventSchema](flat=True, flat_divider="")


def test_model_schema_validation_reports_incompatible_fields() -> None:
    @dataclass
    class Missing:
        missing: str

    @dataclass
    class WrongDtype:
        tags: list[int]

    @dataclass
    class WrongStructure:
        events: list[str]

    with pytest.raises(TypeError, match="is not declared"):
        tp.Builder(DataFrameSchema, Missing)
    with pytest.raises(TypeError, match="uses Python type"):
        tp.Builder(DataFrameSchema, WrongDtype)
    with pytest.raises(TypeError, match="does not match its schema structure"):
        tp.Builder(DataFrameSchema, WrongStructure)


def test_schema_bound_model_rejects_mismatches_during_declaration() -> None:
    with pytest.raises(TypeError, match="is not declared"):

        @tp.model(schema=DataFrameSchema)
        class Missing:
            missing: str

    # Physical Polars names are not used to find model fields.
    with pytest.raises(TypeError, match="is not declared"):

        @tp.model(schema=DataFrameSchema)
        class PhysicalNameOnly:
            requestId: str  # noqa: N815

    with pytest.raises(TypeError, match="uses Python type"):

        @tp.model(schema=DataFrameSchema)
        class WrongType:
            tags: list[int]

    with pytest.raises(TypeError, match="Schema class"):

        @tp.model(schema=object)  # type: ignore[arg-type]
        class WrongSchema:
            request_id: str


def test_schema_rejects_invalid_declarations() -> None:
    with pytest.raises(TypeError, match="cannot be instantiated"):
        DataFrameSchema()

    with pytest.raises(TypeError, match="duplicate Polars name"):

        class Duplicate(tp.Schema):
            first = tp.Column[str](polars_name="same")
            second = tp.Column[str](polars_name="same")

    with pytest.raises(TypeError, match="Cannot infer a Polars dtype"):

        class Unknown(tp.Schema):
            value = tp.Column[complex]()

    with pytest.raises(TypeError, match="Schema class"):

        @dataclass
        class Row:
            value: int

        tp.Builder(object, Row)  # type: ignore[arg-type]


def test_unbound_schema_cannot_serialize_a_row() -> None:
    class UnboundSchema(tp.Schema):
        value = tp.Column[int]()

    @dataclass
    class Row:
        value: int

    with pytest.raises(TypeError, match="has no builder"):
        UnboundSchema.to_frame(Row(1))
    with pytest.raises(TypeError, match="has no builder"):
        UnboundSchema.model_schema(Row)


def test_non_strict_model_and_schema_use_their_field_intersection() -> None:
    class PartialSchema(tp.Schema):
        value = tp.Column[int]()
        schema_only = tp.Column[str]()

    @tp.model(schema=PartialSchema)
    class Partial:
        value: int
        model_only: str = "default"

    row = Partial(1, "local")
    frame = PartialSchema.to_frame(row)

    assert frame.to_dicts() == [{"value": 1}]
    assert PartialSchema.from_frame(Partial, frame) == Partial(1, "default")


def test_non_strict_model_only_fields_require_defaults() -> None:
    class PartialSchema(tp.Schema):
        value = tp.Column[int]()

    with pytest.raises(TypeError, match="must define a default"):

        @tp.model(schema=PartialSchema)
        class Invalid:
            value: int
            model_only: str


def test_strict_model_requires_exact_schema_fields() -> None:
    class ExtraSchema(tp.Schema):
        value = tp.Column[int]()
        schema_only = tp.Column[str]()

    with pytest.raises(TypeError, match="not declared in model"):

        @tp.model(schema=ExtraSchema, strict=True)
        class MissingSchemaField:
            value: int

    class MissingSchema(tp.Schema):
        value = tp.Column[int]()

    with pytest.raises(TypeError, match="not declared in schema"):

        @tp.model(schema=MissingSchema, strict=True)
        class ExtraModelField:
            value: int
            model_only: str = "default"

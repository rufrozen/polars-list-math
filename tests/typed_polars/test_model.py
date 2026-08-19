import importlib
import sys
from dataclasses import dataclass

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


@tp.model
class Suggestion(tp.Model):
    value: str
    corrected_query: str = tp.field(polars_name="correctedQuery")


@tp.model
class Completion(tp.Model):
    prefix: str = tp.field(polars_name="queryPrefix")
    suggestions: list[Suggestion] = tp.field(default_factory=list)


@tp.model
class Row(tp.Model):
    request_id: str = tp.field(polars_name="requestId")
    position: tp.I32
    completion: Completion
    tags: list[str] = tp.field(default_factory=list)


def make_row(request_id: str = "one") -> Row:
    return Row(
        request_id=request_id,
        position=7,
        completion=Completion(
            prefix="по",
            suggestions=[
                Suggestion("поле", "поле"),
                Suggestion("полёт", "полет"),
            ],
        ),
    )


def test_nested_models_use_exact_schema_and_round_trip() -> None:
    row = make_row()

    frame = row.to_frame()

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
    assert Row.from_frame(frame) == row
    assert row.to_dict(by_polars_name=True)["completion"]["queryPrefix"] == "по"
    assert Row.from_dict(row.to_dict(by_polars_name=True), by_polars_name=True) == row


def test_many_rows_are_built_from_physical_tuples() -> None:
    rows = [make_row("one"), make_row("two")]

    frame = Row.to_frame_many(rows)

    assert frame.height == 2
    assert list(Row.iter_frame(frame)) == rows


def test_flat_struct_and_flat_list_struct_round_trip() -> None:
    @tp.model
    class FlatRow(tp.Model):
        completion: Completion = tp.field(flat=True, flat_divider="__")
        suggestions: list[Suggestion] = tp.field(flat=True)

    row = FlatRow(make_row().completion, make_row().completion.suggestions)
    frame = row.to_frame()

    assert frame.schema == pl.Schema(
        {
            "completion__queryPrefix": pl.String,
            "completion__suggestions": pl.List(
                pl.Struct({"value": pl.String, "correctedQuery": pl.String})
            ),
            "suggestions:value": pl.List(pl.String),
            "suggestions:correctedQuery": pl.List(pl.String),
        }
    )
    assert FlatRow.from_frame(frame) == row


def test_top_level_extras_expand_physical_tuple() -> None:
    @tp.model
    class FlexibleRow(tp.Model):
        name: str
        extra: tp.Extras = tp.extras(default_factory=dict)

    rows = [FlexibleRow("one", {"rank": 1}), FlexibleRow("two", {"active": True})]
    frame = FlexibleRow.to_frame_many(rows)

    assert frame.to_dict(as_series=False) == {
        "name": ["one", "two"],
        "active": [None, True],
        "rank": [1, None],
    }
    assert list(FlexibleRow.iter_frame(frame)) == rows


def test_nested_extras_generate_frame_specific_namedtuple() -> None:
    @tp.model
    class Payload(tp.Model):
        title: str
        extra: tp.Extras = tp.extras(default_factory=dict)

    @tp.model
    class FlexibleRow(tp.Model):
        payload: Payload

    rows = [
        FlexibleRow(Payload("one", {"rank": 1})),
        FlexibleRow(Payload("two", {"active": True})),
    ]

    frame = FlexibleRow.to_frame_many(rows)

    assert frame.schema == pl.Schema(
        {"payload": pl.Struct({"title": pl.String, "active": pl.Boolean, "rank": pl.Int64})}
    )
    assert list(FlexibleRow.iter_frame(frame)) == rows


def test_empty_rows_keep_declared_schema() -> None:
    frame = Row.to_frame_many([])

    assert frame.height == 0
    assert frame.schema == Row.polars_schema()


def test_dict_is_stored_as_list_of_key_value_structs() -> None:
    @tp.model
    class DictRow(tp.Model):
        weights: dict[str, float]
        optional: dict[int, str] | None = None

    row = DictRow(weights={"polars": 1.0, "python": 0.5})
    frame = row.to_frame()

    assert frame.schema == pl.Schema(
        {
            "weights": pl.List(pl.Struct({"key": pl.String, "value": pl.Float64})),
            "optional": pl.List(pl.Struct({"key": pl.Int64, "value": pl.String})),
        }
    )
    assert frame.to_dicts() == [
        {
            "weights": [
                {"key": "polars", "value": 1.0},
                {"key": "python", "value": 0.5},
            ],
            "optional": None,
        }
    ]
    assert DictRow.from_frame(frame) == row


def test_columns_build_nested_expressions() -> None:
    frame = make_row().to_frame()

    assert frame.select(Row.columns.completion.fields.prefix.expr()).item() == "по"
    assert frame.select(Row.columns.completion.fields.suggestions.item.value.expr()).to_dict(
        as_series=False
    ) == {"suggestions": [["поле", "полёт"]]}


@pytest.mark.parametrize("alias", ("bad-name", "with.dot", "class", "_private"))
def test_rejects_aliases_that_namedtuple_cannot_represent(alias: str) -> None:
    with pytest.raises(TypeError, match="cannot be represented by NamedTuple"):

        @tp.model
        class Invalid(tp.Model):
            value: int = tp.field(polars_name=alias)


def test_requires_slots_dataclass() -> None:
    @dataclass
    class Invalid(tp.Model):
        value: int

    with pytest.raises(TypeError, match=r"dataclass\(slots=True\)"):
        tp.model(Invalid)


def test_unknown_type_fails_while_module_is_imported() -> None:
    module_name = "tests.typed_polars.unknown_type_module"
    sys.modules.pop(module_name, None)

    with pytest.raises(TypeError, match="Cannot infer a Polars dtype"):
        importlib.import_module(module_name)

    assert module_name not in sys.modules


@pytest.mark.parametrize(
    "annotation",
    (
        list[complex],
        dict[str, complex],
    ),
)
def test_unknown_nested_container_type_fails_during_model_declaration(annotation: object) -> None:
    with pytest.raises(TypeError, match="Cannot infer a Polars dtype"):

        @tp.model
        class InvalidContainer(tp.Model):
            value: annotation  # type: ignore[valid-type]

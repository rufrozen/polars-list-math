from dataclasses import dataclass

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


@tp.model
class Child(tp.Model):
    value: int


@tp.model
class NullableRow(tp.Model):
    child: Child | None
    children: list[Child] | None


@tp.model
class NullableFlatRow(tp.Model):
    child: Child | None = tp.field(flat=True)
    children: list[Child] | None = tp.field(flat=True)


@pytest.mark.parametrize(
    "row",
    (
        NullableRow(None, None),
        NullableRow(Child(1), []),
        NullableRow(Child(1), [Child(2)]),
    ),
)
def test_nullable_nested_values_round_trip(row: NullableRow) -> None:
    assert NullableRow.from_frame(row.to_frame()) == row


@pytest.mark.parametrize(
    "row",
    (
        NullableFlatRow(None, None),
        NullableFlatRow(Child(1), []),
        NullableFlatRow(Child(1), [Child(2)]),
    ),
)
def test_nullable_flat_values_round_trip(row: NullableFlatRow) -> None:
    assert NullableFlatRow.from_frame(row.to_frame()) == row


def test_empty_nested_collections_keep_declared_dtype() -> None:
    frame = NullableRow(Child(1), []).to_frame()

    assert frame.schema["children"] == pl.List(pl.Struct({"value": pl.Int64}))
    assert frame["children"].to_list() == [[]]


def test_duplicate_polars_names_are_rejected_at_model_declaration() -> None:
    with pytest.raises(TypeError, match="duplicate"):

        @tp.model
        class Duplicate(tp.Model):
            first: int = tp.field(polars_name="same")
            second: int = tp.field(polars_name="same")


def test_flat_physical_name_conflict_is_rejected_before_dataframe_construction() -> None:
    @tp.model
    class Conflict(tp.Model):
        child_value: int
        child: Child = tp.field(flat=True, flat_divider="_")

    with pytest.raises(TypeError, match="Physical field name conflict.*child_value"):
        Conflict(1, Child(2)).to_frame()


def test_top_level_extras_cannot_replace_declared_physical_field() -> None:
    @tp.model
    class Flexible(tp.Model):
        value: int
        extra: tp.Extras = tp.extras(default_factory=dict)

    with pytest.raises(TypeError, match="conflicts with declared field 'value'"):
        Flexible(1, {"value": 2}).to_frame()


def test_nested_extras_cannot_replace_declared_struct_field() -> None:
    @tp.model
    class FlexibleChild(tp.Model):
        value: int
        extra: tp.Extras = tp.extras(default_factory=dict)

    @tp.model
    class Parent(tp.Model):
        child: FlexibleChild

    with pytest.raises(TypeError, match="conflicts with declared field 'value'"):
        Parent(FlexibleChild(1, {"value": 2})).to_frame()


def test_multiple_extras_fields_are_rejected() -> None:
    with pytest.raises(TypeError, match="more than one extras"):

        @tp.model
        class Invalid(tp.Model):
            first: tp.Extras = tp.extras(default_factory=dict)
            second: tp.Extras = tp.extras(default_factory=dict)


@pytest.mark.parametrize(
    "options",
    (
        {"flat": True},
        {"flat_divider": ""},
    ),
)
def test_invalid_scalar_flat_options_are_rejected(options: dict[str, object]) -> None:
    with pytest.raises(TypeError):

        @tp.model
        class Invalid(tp.Model):
            value: int = tp.field(**options)  # type: ignore[arg-type]


def test_from_dict_rejects_missing_required_field() -> None:
    with pytest.raises(TypeError, match="required positional argument: 'value'"):
        Child.from_dict({})


def test_from_dict_rejects_unknown_field_without_extras() -> None:
    with pytest.raises(TypeError, match="Unexpected key.*extra"):
        Child.from_dict({"value": 1, "extra": True})


@pytest.mark.parametrize("height", (0, 2))
def test_from_frame_requires_exactly_one_row(height: int) -> None:
    frame = pl.DataFrame({"value": list(range(height))}, schema={"value": pl.Int64})

    with pytest.raises(ValueError, match="Expected exactly one row"):
        Child.from_frame(frame)


def test_flat_list_columns_with_different_lengths_are_rejected() -> None:
    @tp.model
    class Pair(tp.Model):
        left: int
        right: int

    @tp.model
    class Row(tp.Model):
        pairs: list[Pair] = tp.field(flat=True)

    frame = pl.DataFrame(
        {"pairs:left": [[1, 2]], "pairs:right": [[3]]},
        schema={"pairs:left": pl.List(pl.Int64), "pairs:right": pl.List(pl.Int64)},
    )

    with pytest.raises(TypeError, match="different lengths"):
        Row.from_frame(frame)


def test_explicit_slots_dataclass_options_are_preserved() -> None:
    @tp.model
    @dataclass(slots=True, frozen=True, kw_only=True)
    class Frozen(tp.Model):
        value: int

    row = Frozen(value=1)

    assert Frozen.from_frame(row.to_frame()) == row
    with pytest.raises(AttributeError):
        row.value = 2  # type: ignore[misc]


def test_custom_dtype_accepts_storage_compatible_custom_value() -> None:
    class Identifier(str):
        pass

    @tp.model
    class CustomRow(tp.Model):
        value: Identifier = tp.field(dtype=pl.String)

    frame = CustomRow(Identifier("one")).to_frame()

    assert frame.schema == pl.Schema({"value": pl.String})
    assert frame["value"].to_list() == ["one"]


def test_strict_frame_schema_rejects_missing_extra_and_wrong_dtype() -> None:
    valid = Child(1).to_frame()
    Child.assert_frame_schema(valid)

    invalid_frames = (
        pl.DataFrame({"other": [1]}),
        pl.DataFrame({"value": ["one"]}),
        valid.with_columns(pl.lit(True).alias("extra")),
    )
    for frame in invalid_frames:
        with pytest.raises(TypeError, match="Unexpected DataFrame schema"):
            Child.from_frame(frame, strict_schema=True)


def test_non_strict_frame_ignores_unknown_top_level_and_struct_fields() -> None:
    @tp.model
    class Parent(tp.Model):
        child: Child

    frame = pl.DataFrame(
        {
            "child": [{"value": 1, "unknownNested": "ignored"}],
            "unknownTop": [True],
        }
    )

    assert Parent.from_frame(frame, strict_schema=False) == Parent(Child(1))
    with pytest.raises(TypeError, match="Unexpected DataFrame schema"):
        Parent.from_frame(frame, strict_schema=True)


def test_non_strict_frame_ignores_unknown_list_struct_fields() -> None:
    @tp.model
    class Parent(tp.Model):
        children: list[Child]

    frame = pl.DataFrame(
        {
            "children": [
                [
                    {"value": 1, "unknown": "first"},
                    {"value": 2, "unknown": "second"},
                ]
            ]
        }
    )

    assert Parent.from_frame(frame) == Parent([Child(1), Child(2)])


def test_non_strict_frame_ignores_unknown_flat_struct_columns() -> None:
    @tp.model
    class Pair(tp.Model):
        left: int
        right: int

    @tp.model
    class FlatRow(tp.Model):
        pair: Pair = tp.field(flat=True)
        pairs: list[Pair] = tp.field(flat=True)

    frame = (
        FlatRow(Pair(1, 2), [Pair(3, 4)])
        .to_frame()
        .with_columns(
            pl.lit("top").alias("unknownTop"),
            pl.lit("nested").alias("pair:unknown"),
            pl.lit(["nested-list"]).alias("pairs:unknown"),
        )
    )

    assert FlatRow.from_frame(frame, strict_schema=False) == FlatRow(Pair(1, 2), [Pair(3, 4)])
    with pytest.raises(TypeError, match="Unexpected DataFrame schema"):
        FlatRow.from_frame(frame, strict_schema=True)


def test_non_strict_frame_still_captures_unknown_values_when_extras_exist() -> None:
    @tp.model
    class FlexibleChild(tp.Model):
        value: int
        extra: tp.Extras = tp.extras(default_factory=dict)

    @tp.model
    class FlexibleParent(tp.Model):
        child: FlexibleChild
        extra: tp.Extras = tp.extras(default_factory=dict)

    frame = pl.DataFrame(
        {
            "child": [{"value": 1, "nestedExtra": 2}],
            "topExtra": [3],
        }
    )
    row = FlexibleParent.from_frame(frame, strict_schema=False)

    assert row.child.extra == {"nestedExtra": 2}
    assert row.extra == {"topExtra": 3}


def test_strict_schema_reports_dynamic_extras_as_unsupported() -> None:
    @tp.model
    class Flexible(tp.Model):
        value: int
        extra: tp.Extras = tp.extras(default_factory=dict)

    frame = Flexible(1, {"dynamic": True}).to_frame()

    with pytest.raises(TypeError, match="unavailable for models with Extras"):
        Flexible.from_frame(frame, strict_schema=True)


def test_single_model_inheritance_adds_slots_fields() -> None:
    @tp.model
    class Base(tp.Model):
        inherited: str

    @tp.model
    class Derived(Base):
        own: int

    row = Derived("base", 2)

    assert not hasattr(row, "__dict__")
    assert Derived.from_frame(row.to_frame()) == row


def test_column_expression_crosses_list_struct_then_struct() -> None:
    @tp.model
    class Details(tp.Model):
        code: int

    @tp.model
    class Item(tp.Model):
        details: Details

    @tp.model
    class Row(tp.Model):
        items: list[Item]

    frame = Row([Item(Details(7)), Item(Details(8))]).to_frame()
    column = Row.columns.items.item.details.fields.code

    assert frame.select(column.expr()).to_dict(as_series=False) == {"items": [[7, 8]]}


def test_flat_column_expressions_use_physical_columns() -> None:
    @tp.model
    class Pair(tp.Model):
        name: str
        value: int

    @tp.model
    class Row(tp.Model):
        pair: Pair = tp.field(flat=True, flat_divider="__")
        pairs: list[Pair] = tp.field(flat=True, flat_divider="__")

    frame = Row(Pair("one", 1), [Pair("two", 2)]).to_frame()

    assert frame.select(Row.columns.pair.fields.value.expr()).item() == 1
    assert frame.select(Row.columns.pairs.item.name.expr()).to_dict(as_series=False) == {
        "pairs__name": [["two"]]
    }


def test_wrong_nested_runtime_values_raise_contextual_type_errors() -> None:
    with pytest.raises(TypeError, match="expected Child, got dict"):
        NullableRow({"value": 1}, []).to_frame()  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a list of Child"):
        NullableRow(None, [{"value": 1}]).to_frame()  # type: ignore[list-item]


def test_from_dict_rejects_wrong_nested_shapes() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        NullableRow.from_dict({"child": 1, "children": []})

    with pytest.raises(TypeError, match="must be a list of mappings"):
        NullableRow.from_dict({"child": None, "children": [1]})


def test_default_factories_create_independent_values() -> None:
    @tp.model
    class Defaults(tp.Model):
        values: list[int] = tp.field(default_factory=list)

    first = Defaults()
    second = Defaults()
    first.values.append(1)

    assert second.values == []


def test_extras_infer_nullable_and_all_null_dtypes() -> None:
    @tp.model
    class Flexible(tp.Model):
        value: int
        extra: tp.Extras = tp.extras(default_factory=dict)

    frame = Flexible.to_frame_many(
        [
            Flexible(1, {"rank": None, "empty": None}),
            Flexible(2, {"rank": 2}),
        ]
    )

    assert frame.schema == pl.Schema({"value": pl.Int64, "empty": pl.Null, "rank": pl.Int64})
    assert frame.to_dict(as_series=False) == {
        "value": [1, 2],
        "empty": [None, None],
        "rank": [None, 2],
    }


def test_extra_name_must_be_namedtuple_compatible() -> None:
    @tp.model
    class Flexible(tp.Model):
        extra: tp.Extras = tp.extras(default_factory=dict)

    with pytest.raises(TypeError, match="Polars name.*cannot be represented"):
        Flexible({"bad-name": 1}).to_frame()


def test_dict_rejects_non_mapping_runtime_value() -> None:
    @tp.model
    class DictRow(tp.Model):
        values: dict[str, int]

    with pytest.raises(TypeError, match="Field 'values' must be a mapping"):
        DictRow([("one", 1)]).to_frame()  # type: ignore[arg-type]


def test_empty_and_nullable_dicts_round_trip() -> None:
    @tp.model
    class DictRow(tp.Model):
        values: dict[str, int] | None

    rows = [DictRow(None), DictRow({}), DictRow({"one": 1})]
    frame = DictRow.to_frame_many(rows)

    assert frame["values"].to_list() == [None, [], [{"key": "one", "value": 1}]]
    assert list(DictRow.iter_frame(frame)) == rows

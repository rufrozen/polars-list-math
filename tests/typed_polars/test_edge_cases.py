from dataclasses import field

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


class ChildSchema(tp.Schema):
    value = tp.Column[int]()


class NullableSchema(tp.Schema):
    child = tp.Struct[ChildSchema]()
    children = tp.ListStruct[ChildSchema]()


@tp.model(schema=ChildSchema)
class Child:
    value: int


@tp.model(schema=NullableSchema)
class NullableRow:
    child: Child | None
    children: list[Child] | None


@pytest.mark.parametrize(
    "row", (NullableRow(None, None), NullableRow(Child(1), []), NullableRow(Child(1), [Child(2)]))
)
def test_nullable_nested_values_round_trip(row: NullableRow) -> None:
    assert NullableSchema.from_frame(NullableRow, NullableSchema.to_frame(row)) == row


def test_empty_nested_collections_keep_schema_dtype() -> None:
    frame = NullableSchema.to_frame(NullableRow(Child(1), []))
    assert frame.schema["children"] == pl.List(pl.Struct({"value": pl.Int64}))


def test_dataclasses_field_owns_defaults() -> None:
    class ValuesSchema(tp.Schema):
        values = tp.Column[list[int]]()

    @tp.model(schema=ValuesSchema)
    class Values:
        values: list[int] = field(default_factory=list)

    assert ValuesSchema.from_frame(Values, ValuesSchema.to_frame(Values())) == Values()


def test_runtime_nested_values_are_checked() -> None:
    with pytest.raises(TypeError, match="Expected Child"):
        NullableSchema.to_frame(NullableRow({"value": 1}, []))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Expected Child"):
        NullableSchema.to_frame(NullableRow(None, [{"value": 1}]))  # type: ignore[list-item]


def test_non_strict_frame_ignores_unknown_fields() -> None:
    frame = pl.DataFrame(
        {"child": [{"value": 1, "unknown": True}], "children": [[]], "unknown": [1]}
    )
    assert NullableSchema.from_frame(NullableRow, frame) == NullableRow(Child(1), [])
    with pytest.raises(TypeError, match="Unexpected DataFrame schema"):
        NullableSchema.from_frame(NullableRow, frame, strict_schema=True)


def test_from_frame_requires_one_row_and_dict_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="exactly one row"):
        ChildSchema.from_frame(Child, pl.DataFrame({"value": []}, schema={"value": pl.Int64}))
    with pytest.raises(TypeError, match="Unexpected key"):
        ChildSchema.from_dict(Child, {"value": 1, "extra": True})

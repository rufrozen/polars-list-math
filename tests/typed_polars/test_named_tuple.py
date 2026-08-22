from collections import namedtuple
from dataclasses import dataclass, is_dataclass
from typing import NamedTuple, assert_type

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


class ChildSchema(tp.Schema):
    value = tp.Column[int]()
    label = tp.Column[str]()


class TupleRowSchema(tp.Schema):
    request_id = tp.Column[str](polars_name="requestId")
    child = tp.Struct[ChildSchema]()
    children = tp.ListStruct[ChildSchema]()


class TupleChild(NamedTuple):
    value: int
    label: str


@tp.model(schema=TupleRowSchema, strict=True)
class TupleRow(NamedTuple):
    request_id: str
    child: TupleChild
    children: list[TupleChild]


def test_named_tuple_root_and_nested_values_round_trip() -> None:
    row = TupleRow("one", TupleChild(1, "first"), [TupleChild(2, "second")])

    frame = TupleRowSchema.to_frame(row)
    restored = TupleRowSchema.from_frame(TupleRow, frame, strict_schema=True)

    assert frame.schema == pl.Schema(
        {
            "requestId": pl.String,
            "child": pl.Struct({"value": pl.Int64, "label": pl.String}),
            "children": pl.List(pl.Struct({"value": pl.Int64, "label": pl.String})),
        }
    )
    assert restored == row
    assert type(restored) is TupleRow
    assert isinstance(restored, tuple)
    assert not is_dataclass(TupleRow)
    assert (
        TupleRowSchema.from_dict(
            TupleRow,
            TupleRowSchema.to_dict(row, by_polars_name=True),
            by_polars_name=True,
        )
        == row
    )
    assert_type(restored, TupleRow)


def test_dataclass_and_named_tuple_models_can_be_mixed_recursively() -> None:
    class DataRootSchema(tp.Schema):
        child = tp.FlatStruct[ChildSchema]()

    @tp.model(schema=DataRootSchema)
    @dataclass
    class DataRoot:
        child: TupleChild

    @dataclass
    class DataChild:
        value: int
        label: str

    class NamedRootSchema(tp.Schema):
        children = tp.FlatListStruct[ChildSchema]()

    @tp.model(schema=NamedRootSchema)
    class NamedRoot(NamedTuple):
        children: list[DataChild]

    data_row = DataRoot(TupleChild(1, "tuple"))
    named_row = NamedRoot([DataChild(2, "dataclass")])

    assert (
        DataRootSchema.from_frame(
            DataRoot,
            DataRootSchema.to_frame(data_row),
        )
        == data_row
    )
    assert (
        NamedRootSchema.from_frame(
            NamedRoot,
            NamedRootSchema.to_frame(named_row),
        )
        == named_row
    )


def test_named_tuple_model_only_fields_use_field_defaults() -> None:
    class PartialSchema(tp.Schema):
        value = tp.Column[int]()

    @tp.model(schema=PartialSchema)
    class Partial(NamedTuple):
        value: int
        local: str = "fallback"

    row = Partial(1, "not serialized")

    assert PartialSchema.to_frame(row).to_dicts() == [{"value": 1}]
    assert PartialSchema.from_frame(Partial, PartialSchema.to_frame(row)) == Partial(
        1,
        "fallback",
    )

    with pytest.raises(TypeError, match="must define a default"):

        @tp.model(schema=PartialSchema)
        class MissingDefault(NamedTuple):
            value: int
            local: str


def test_untyped_named_tuple_is_rejected_at_root_and_nested_levels() -> None:
    UntypedChild = namedtuple("UntypedChild", ["value", "label"])

    with pytest.raises(TypeError, match="typed NamedTuple"):
        tp.model(schema=ChildSchema)(UntypedChild)

    class InvalidRoot(NamedTuple):
        child: UntypedChild

    class InvalidSchema(tp.Schema):
        child = tp.Struct[ChildSchema]()

    with pytest.raises(TypeError, match="typed NamedTuple"):
        tp.model(schema=InvalidSchema)(InvalidRoot)

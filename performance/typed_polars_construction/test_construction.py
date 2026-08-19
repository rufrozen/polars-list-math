from __future__ import annotations

import os
from collections.abc import Callable
from time import perf_counter
from typing import Any, NamedTuple

import polars as pl
import polars_list_math.typed_polars as tp
import pytest

DEFAULT_ROW_COUNT = 10_000
ROW_COUNT = int(os.environ.get("POLARS_TYPED_ROWS", DEFAULT_ROW_COUNT))


@tp.model
class Address(tp.Model):
    city: str
    zip_code: int = tp.field(polars_name="zipCode")


@tp.model
class Item(tp.Model):
    item_id: int = tp.field(polars_name="itemId")
    label: str
    score: tp.F32


@tp.model
class Payload(tp.Model):
    title: str
    address: Address
    items: list[Item] = tp.field(default_factory=list)


@tp.model
class Metrics(tp.Model):
    views: int
    quality: tp.F32


@tp.model
class Feature(tp.Model):
    code: str
    weight: tp.F32


@tp.model
class TypedRow(tp.Model):
    row_id: int = tp.field(polars_name="rowId")
    name: str
    active: bool
    count: int
    ratio: float
    category: str
    status: str
    priority: int
    amount: float
    enabled: bool
    payload: Payload
    metrics: Metrics = tp.field(flat=True, flat_divider="__")
    features: list[Feature] = tp.field(flat=True, flat_divider="__")
    tags: list[str] = tp.field(default_factory=list)
    values: list[int] = tp.field(default_factory=list)
    weights: dict[str, float] = tp.field(default_factory=dict)
    note: str | None = None


class AddressTuple(NamedTuple):
    city: str
    zipCode: int  # noqa: N815 - must match the physical Polars alias


class ItemTuple(NamedTuple):
    itemId: int  # noqa: N815 - must match the physical Polars alias
    label: str
    score: float


class PayloadTuple(NamedTuple):
    title: str
    address: AddressTuple
    items: list[ItemTuple]


class KeyValueTuple(NamedTuple):
    key: str
    value: float


class ReadyNamedTupleRow(NamedTuple):
    rowId: int  # noqa: N815 - must match the physical Polars alias
    name: str
    active: bool
    count: int
    ratio: float
    category: str
    status: str
    priority: int
    amount: float
    enabled: bool
    payload: PayloadTuple
    metrics__views: int
    metrics__quality: float
    features__code: list[str]
    features__weight: list[float]
    tags: list[str]
    values: list[int]
    weights: list[KeyValueTuple]
    note: str | None


PAYLOAD_DTYPE = pl.Struct(
    {
        "title": pl.String,
        "address": pl.Struct({"city": pl.String, "zipCode": pl.Int64}),
        "items": pl.List(pl.Struct({"itemId": pl.Int64, "label": pl.String, "score": pl.Float32})),
    }
)
EXPECTED_SCHEMA = pl.Schema(
    {
        "rowId": pl.Int64,
        "name": pl.String,
        "active": pl.Boolean,
        "count": pl.Int64,
        "ratio": pl.Float64,
        "category": pl.String,
        "status": pl.String,
        "priority": pl.Int64,
        "amount": pl.Float64,
        "enabled": pl.Boolean,
        "payload": PAYLOAD_DTYPE,
        "metrics__views": pl.Int64,
        "metrics__quality": pl.Float32,
        "features__code": pl.List(pl.String),
        "features__weight": pl.List(pl.Float32),
        "tags": pl.List(pl.String),
        "values": pl.List(pl.Int64),
        "weights": pl.List(pl.Struct({"key": pl.String, "value": pl.Float64})),
        "note": pl.String,
    }
)


def make_typed_row(index: int) -> TypedRow:
    return TypedRow(
        row_id=index,
        name=f"row-{index}",
        active=index % 2 == 0,
        count=index * 2,
        ratio=index / 10,
        category=f"category-{index % 5}",
        status="ready" if index % 2 == 0 else "pending",
        priority=index % 10,
        amount=index * 1.25,
        enabled=index % 3 != 0,
        payload=Payload(
            title=f"payload-{index}",
            address=Address(city=f"city-{index % 10}", zip_code=100_000 + index),
            items=[
                Item(index * 2, "first", 0.5),
                Item(index * 2 + 1, "second", 0.75),
            ],
        ),
        metrics=Metrics(views=100 + index, quality=0.5),
        features=[
            Feature("fresh", 0.25),
            Feature("popular", 0.75),
        ],
        tags=[f"tag-{index % 3}", f"tag-{(index + 1) % 3}"],
        values=[index, index + 1, index + 2],
        weights={"primary": float(index), "secondary": index + 0.5},
        note=None if index % 3 == 0 else f"note-{index}",
    )


def make_namedtuple_row(index: int) -> ReadyNamedTupleRow:
    return ReadyNamedTupleRow(
        rowId=index,
        name=f"row-{index}",
        active=index % 2 == 0,
        count=index * 2,
        ratio=index / 10,
        category=f"category-{index % 5}",
        status="ready" if index % 2 == 0 else "pending",
        priority=index % 10,
        amount=index * 1.25,
        enabled=index % 3 != 0,
        payload=PayloadTuple(
            title=f"payload-{index}",
            address=AddressTuple(city=f"city-{index % 10}", zipCode=100_000 + index),
            items=[
                ItemTuple(index * 2, "first", 0.5),
                ItemTuple(index * 2 + 1, "second", 0.75),
            ],
        ),
        metrics__views=100 + index,
        metrics__quality=0.5,
        features__code=["fresh", "popular"],
        features__weight=[0.25, 0.75],
        tags=[f"tag-{index % 3}", f"tag-{(index + 1) % 3}"],
        values=[index, index + 1, index + 2],
        weights=[
            KeyValueTuple("primary", float(index)),
            KeyValueTuple("secondary", index + 0.5),
        ],
        note=None if index % 3 == 0 else f"note-{index}",
    )


def build_typed(rows: list[TypedRow]) -> pl.DataFrame:
    return TypedRow.to_frame_many(rows)


def build_ready_namedtuple(rows: list[ReadyNamedTupleRow]) -> pl.DataFrame:
    return pl.DataFrame(rows, orient="row", schema=EXPECTED_SCHEMA)


CASES = (
    ("typed_polars", make_typed_row, build_typed),
    ("ready_nested_namedtuple", make_namedtuple_row, build_ready_namedtuple),
)


@pytest.mark.parametrize(("name", "row_factory", "builder"), CASES, ids=lambda value: value)
@pytest.mark.parametrize("row_count", (1, ROW_COUNT), ids=("one_row", "n_rows"))
def test_construction_performance(
    name: str,
    row_factory: Callable[[int], Any],
    builder: Callable[[Any], pl.DataFrame],
    row_count: int,
) -> None:
    rows = [row_factory(index) for index in range(row_count)]

    started_at = perf_counter()
    frame = builder(rows)
    elapsed = perf_counter() - started_at

    print(f"{name:>24} rows={row_count:>7}: {elapsed:.4f}s")
    assert frame.shape == (row_count, 19)
    assert frame.schema == EXPECTED_SCHEMA
    assert frame["rowId"].to_list() == list(range(row_count))
    assert frame["payload"].struct.field("address").struct.field("city")[0] == "city-0"
    assert frame["metrics__views"][0] == 100
    assert frame["features__code"].to_list()[0] == ["fresh", "popular"]

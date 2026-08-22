"""Flatten nested models and runtime dictionary keys through their schema."""

from dataclasses import dataclass

import polars_list_math.typed_polars as tp


class MetricsSchema(tp.Schema):
    views = tp.Column[int]()
    conversion_rate = tp.Column[tp.F32](polars_name="conversionRate")
    sources = tp.FlatDict[int](divider="__")


class RowSchema(tp.Schema):
    request_id = tp.Column[str](polars_name="requestId")
    metrics = tp.FlatStruct[MetricsSchema](divider="__")


@dataclass
class Metrics:
    views: int
    conversion_rate: float
    sources: dict[str, int]


@tp.model(schema=RowSchema)
@dataclass
class Row:
    request_id: str
    metrics: Metrics


row = Row("request-1", Metrics(120, 0.5, {"organic": 100, "paid": 20}))
context = tp.Context().bind(
    RowSchema.metrics.fields.sources,
    ["organic", "paid"],
)
frame = RowSchema.to_frame(row, context=context)

assert frame.columns == [
    "requestId",
    "metrics__views",
    "metrics__conversionRate",
    "metrics__sources__organic",
    "metrics__sources__paid",
]
assert RowSchema.from_frame(Row, frame) == row
assert frame.select(RowSchema.metrics.fields.sources.key_expr("paid")).to_dicts() == [
    {"metrics__sources__paid": 20}
]
print(frame)

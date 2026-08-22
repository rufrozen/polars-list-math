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
    history = tp.FlatListStruct[MetricsSchema](divider="__")


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
    history: list[Metrics]


row = Row(
    "request-1",
    Metrics(120, 0.5, {"organic": 100, "paid": 20}),
    [
        Metrics(80, 0.5, {"organic": 70, "paid": 10}),
        Metrics(40, 0.25, {"organic": 40}),
    ],
)
context = (
    tp.Context()
    .bind(RowSchema.metrics.fields.sources, ["organic", "paid"])
    .bind(RowSchema.history.item.sources, ["organic", "paid"])
)
frame = RowSchema.to_frame(row, context=context)

assert frame.columns == [
    "requestId",
    "metrics__views",
    "metrics__conversionRate",
    "metrics__sources__organic",
    "metrics__sources__paid",
    "history__views",
    "history__conversionRate",
    "history__sources__organic",
    "history__sources__paid",
]
assert RowSchema.from_frame(Row, frame) == row
assert frame.select(
    RowSchema.metrics.fields.sources.key_expr("paid"),
    RowSchema.history.item.conversion_rate.expr(),
    RowSchema.history.item.sources.key_expr("paid"),
).to_dicts() == [
    {
        "metrics__sources__paid": 20,
        "history__conversionRate": [0.5, 0.25],
        "history__sources__paid": [10, None],
    }
]
print(frame)

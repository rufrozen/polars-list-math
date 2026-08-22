"""Flatten nested models and runtime dictionary keys through their schema."""

from dataclasses import dataclass

import polars_list_math.typed_polars as tp


class MetricsSchema(tp.Schema):
    views = tp.Column[int]()
    conversion_rate = tp.Column[tp.F32](polars_name="conversionRate")
    sources = tp.FlatDict[int](divider="__")
    scores = tp.FlatTuple[float](divider="__")


class RowSchema(tp.Schema):
    request_id = tp.Column[str](polars_name="requestId")
    metrics = tp.FlatStruct[MetricsSchema](divider="__")
    history = tp.FlatListStruct[MetricsSchema](divider="__")


@dataclass
class Metrics:
    views: int
    conversion_rate: float
    sources: dict[str, int]
    scores: tuple[float, ...]


@tp.model(schema=RowSchema)
@dataclass
class Row:
    request_id: str
    metrics: Metrics
    history: list[Metrics]


row = Row(
    "request-1",
    Metrics(120, 0.5, {"organic": 100, "paid": 20}, (0.8, 0.6)),
    [
        Metrics(80, 0.5, {"organic": 70, "paid": 10}, (0.7, 0.5)),
        Metrics(40, 0.25, {"organic": 40}, (0.9, 0.4)),
    ],
)
context = (
    tp.Context()
    .bind(RowSchema.metrics.fields.sources, ["organic", "paid"])
    .bind(RowSchema.metrics.fields.scores, ["precision", "recall"])
    .bind(RowSchema.history.item.sources, ["organic", "paid"])
    .bind(RowSchema.history.item.scores, ["precision", "recall"])
)
frame = RowSchema.to_frame(row, context=context)

assert frame.columns == [
    "requestId",
    "metrics__views",
    "metrics__conversionRate",
    "metrics__sources__organic",
    "metrics__sources__paid",
    "metrics__scores__precision",
    "metrics__scores__recall",
    "history__views",
    "history__conversionRate",
    "history__sources__organic",
    "history__sources__paid",
    "history__scores__precision",
    "history__scores__recall",
]
assert RowSchema.from_frame(Row, frame) == row
assert frame.select(
    RowSchema.metrics.fields.sources.key_expr("paid"),
    RowSchema.metrics.fields.scores.key_expr("recall"),
    RowSchema.history.item.conversion_rate.expr(),
    RowSchema.history.item.sources.key_expr("paid"),
).to_dicts() == [
    {
        "metrics__sources__paid": 20,
        "metrics__scores__recall": 0.6,
        "history__conversionRate": [0.5, 0.25],
        "history__sources__paid": [10, None],
    }
]
print(frame)

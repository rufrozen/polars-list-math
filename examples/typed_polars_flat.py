"""Flatten nested models through their schema declaration."""

from dataclasses import dataclass

import polars_list_math.typed_polars as tp


class MetricsSchema(tp.Schema):
    views = tp.Column[int]()
    conversion_rate = tp.Column[tp.F32](polars_name="conversionRate")


class RowSchema(tp.Schema):
    request_id = tp.Column[str](polars_name="requestId")
    metrics = tp.Struct[MetricsSchema](flat=True, flat_divider="__")


@dataclass
class Metrics:
    views: int
    conversion_rate: float


@tp.model(schema=RowSchema)
@dataclass
class Row:
    request_id: str
    metrics: Metrics


row = Row("request-1", Metrics(120, 0.5))
frame = RowSchema.to_frame(row)
assert frame.columns == ["requestId", "metrics__views", "metrics__conversionRate"]
assert RowSchema.from_frame(Row, frame) == row
print(frame)

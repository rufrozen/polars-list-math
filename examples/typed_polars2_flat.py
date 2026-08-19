"""Flatten nested typed_polars2 models into physical Polars columns."""

import polars_list_math.typed_polars2 as tp


@tp.model
class Metrics(tp.Model):
    views: int
    conversion_rate: tp.F32 = tp.field(polars_name="conversionRate")


@tp.model
class Item(tp.Model):
    name: str
    score: tp.F32


@tp.model
class SearchResult(tp.Model):
    request_id: str = tp.field(polars_name="requestId")
    metrics: Metrics = tp.field(flat=True, flat_divider="__")
    items: list[Item] = tp.field(flat=True, flat_divider="__")


rows = [
    SearchResult(
        request_id="request-1",
        metrics=Metrics(views=120, conversion_rate=0.5),
        items=[Item(name="polars", score=0.75), Item(name="python", score=0.5)],
    ),
    SearchResult(
        request_id="request-2",
        metrics=Metrics(views=80, conversion_rate=0.25),
        items=[Item(name="rust", score=1.0)],
    ),
]

frame = SearchResult.to_frame_many(rows)
print(frame)
print(frame.schema)

assert frame.columns == [
    "requestId",
    "metrics__views",
    "metrics__conversionRate",
    "items__name",
    "items__score",
]
assert frame["metrics__views"].to_list() == [120, 80]
assert frame["items__name"].to_list() == [["polars", "python"], ["rust"]]

# Flat physical columns are assembled back into nested public dataclasses.
restored = list(SearchResult.iter_frame(frame))
assert restored == rows
print(restored)
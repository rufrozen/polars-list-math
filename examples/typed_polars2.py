"""Build Polars frames from tuple-oriented slots dataclasses."""

from datetime import datetime

import polars_list_math.typed_polars2 as tp


@tp.model
class Suggestion(tp.Model):
    value: str
    score: tp.F32
    corrected_query: str | None = tp.field(
        default=None,
        polars_name="correctedQuery",
    )


@tp.model
class SearchResult(tp.Model):
    request_id: str = tp.field(polars_name="requestId")
    weights: dict[str, float]
    suggestions: list[Suggestion] = tp.field(default_factory=list)
    labels: dict[int, str] = tp.field(default_factory=dict)
    extra: tp.Extras = tp.extras(default_factory=dict)


@tp.model
class Event(tp.Model):
    timestamp: tp.TimestampMs
    search: SearchResult
    tags: list[str] = tp.field(default_factory=list)


rows = [
    Event(
        # TimestampMs intentionally declares a timezone-naive Polars dtype.
        timestamp=datetime(2026, 8, 19, 12),  # noqa: DTZ001
        search=SearchResult(
            request_id="request-1",
            weights={"polars": 1.0, "python": 0.5},
            suggestions=[
                Suggestion("polars", 0.50),
                Suggestion("python", 0.75, corrected_query="python language"),
            ],
            labels={1: "primary", 2: "secondary"},
            extra={"source": "web"},
        ),
        tags=["typed", "nested"],
    ),
    Event(
        timestamp=datetime(2026, 8, 19, 13),  # noqa: DTZ001
        search=SearchResult(
            request_id="request-2",
            weights={"rust": 0.75},
            suggestions=[Suggestion("rust", 0.25)],
            extra={"page": 2},
        ),
    ),
]

frame = Event.to_frame_many(rows)
print(frame)
print(frame.schema)

# dict[K, V] is stored as List[Struct[key: K, value: V]].
assert frame["search"].struct.field("weights").to_list()[0] == [
    {"key": "polars", "value": 1.0},
    {"key": "python", "value": 0.5},
]

# Class-level columns are separate from dataclass instance attributes.
scores = Event.columns.search.fields.suggestions.item.score
print(frame.select(scores.expr()))

# Polars Struct values are converted back into the public slots dataclasses.
restored = list(Event.iter_frame(frame))
assert restored == rows
print(restored[0].search.request_id)
print(restored[0].search.weights)

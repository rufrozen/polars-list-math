"""Build nested and flat Polars frames from typed schemas."""

from datetime import datetime

import polars_list_math.typed_polars as tp


class Suggestion(tp.Schema):
    value = tp.Field[str]()
    score = tp.Field[tp.F32]()


class SearchResult(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    suggestions = tp.ListStruct[Suggestion]()


class Event(tp.Schema):
    timestamp = tp.Field[tp.TimestampMs]()
    result = tp.Struct[SearchResult](flat_alias="search")


event = Event(
    timestamp=datetime(2026, 8, 16, 12),
    result=SearchResult(
        request_id="request-1",
        suggestions=[
            Suggestion(value="polars", score=0.95),
            Suggestion(value="python", score=0.80),
        ],
    ),
)

print(event.to_frame())
print(event.to_flat_frame())
print(Event.result.fields.suggestions.item.score.nested_expr())

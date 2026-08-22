"""Exercise every Python type and explicit dtype supported by typed_polars."""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import polars_list_math.typed_polars as tp


class ProfileSchema(tp.Schema):
    name = tp.Column[str]()
    age = tp.Column[int]()


class SuggestionSchema(tp.Schema):
    value = tp.Column[str]()
    score = tp.Column[tp.F32]()


class AllTypesSchema(tp.Schema):
    # Inferred Python scalar types.
    request_id = tp.Column[str](polars_name="requestId")
    integer = tp.Column[int]()
    floating = tp.Column[float]()
    active = tp.Column[bool]()
    binary = tp.Column[bytes]()
    calendar_date = tp.Column[date]()
    timestamp = tp.Column[datetime]()
    duration = tp.Column[timedelta]()

    # Explicit-width integer and floating-point types.
    i8 = tp.Column[tp.I8]()
    i16 = tp.Column[tp.I16]()
    i32 = tp.Column[tp.I32]()
    i64 = tp.Column[tp.I64]()
    u8 = tp.Column[tp.U8]()
    u16 = tp.Column[tp.U16]()
    u32 = tp.Column[tp.U32]()
    u64 = tp.Column[tp.U64]()
    f32 = tp.Column[tp.F32]()
    f64 = tp.Column[tp.F64]()

    # Explicit temporal resolutions.
    timestamp_ms = tp.Column[tp.TimestampMs]()
    timestamp_us = tp.Column[tp.TimestampUs]()
    timestamp_ns = tp.Column[tp.TimestampNs]()
    duration_ms = tp.Column[tp.DurationMs]()
    duration_us = tp.Column[tp.DurationUs]()
    duration_ns = tp.Column[tp.DurationNs]()

    # Nested dataclass, nullable, and container types.
    profile = tp.Struct[ProfileSchema]()
    optional_text = tp.Column[str | None](polars_name="optionalText")
    tags = tp.Column[list[str]]()
    weights = tp.Column[dict[str, float]]()
    dynamic_metrics = tp.FlatDict[float]()
    suggestions = tp.ListStruct[SuggestionSchema]()


@dataclass
class Profile:
    name: str
    age: int


@dataclass
class Suggestion:
    value: str
    score: float


@tp.model(schema=AllTypesSchema, strict=True)
@dataclass
class AllTypesRow:
    request_id: str
    integer: int
    floating: float
    active: bool
    binary: bytes
    calendar_date: date
    timestamp: datetime
    duration: timedelta
    i8: int
    i16: int
    i32: int
    i64: int
    u8: int
    u16: int
    u32: int
    u64: int
    f32: float
    f64: float
    timestamp_ms: datetime
    timestamp_us: datetime
    timestamp_ns: datetime
    duration_ms: timedelta
    duration_us: timedelta
    duration_ns: timedelta
    profile: Profile
    optional_text: str | None = None
    tags: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    dynamic_metrics: dict[str, float] = field(default_factory=dict)
    suggestions: list[Suggestion] = field(default_factory=list)


moment = datetime(2026, 8, 22, 12)  # noqa: DTZ001
elapsed = timedelta(seconds=5)
row = AllTypesRow(
    request_id="request-1",
    integer=-1,
    floating=0.5,
    active=True,
    binary=b"typed-polars",
    calendar_date=date(2026, 8, 22),
    timestamp=moment,
    duration=elapsed,
    i8=-8,
    i16=-16,
    i32=-32,
    i64=-64,
    u8=8,
    u16=16,
    u32=32,
    u64=64,
    f32=0.25,
    f64=0.5,
    timestamp_ms=moment,
    timestamp_us=moment,
    timestamp_ns=moment,
    duration_ms=elapsed,
    duration_us=elapsed,
    duration_ns=elapsed,
    profile=Profile("Polars", 10),
    tags=["typed", "schema"],
    weights={"quality": 1.0},
    dynamic_metrics={"views": 100.0, "conversion": 0.5},
    suggestions=[Suggestion("polars", 0.75)],
)

context = tp.Context().bind(
    AllTypesSchema.dynamic_metrics,
    ["views", "conversion"],
)
frame = AllTypesSchema.to_frame(row, context=context)
restored = AllTypesSchema.from_frame(AllTypesRow, frame)

assert frame.schema == AllTypesSchema.polars_schema(context=context)
assert restored == row

selected = frame.select(
    AllTypesSchema.request_id.expr(),
    AllTypesSchema.profile.fields.name.expr().alias("profileName"),
    AllTypesSchema.suggestions.item.score.expr().alias("suggestionScores"),
    AllTypesSchema.dynamic_metrics.key_expr("conversion"),
)
assert selected.to_dicts() == [
    {
        "requestId": "request-1",
        "profileName": "Polars",
        "suggestionScores": [0.75],
        "dynamic_metrics_conversion": 0.5,
    }
]

print(frame.schema)
print(selected)

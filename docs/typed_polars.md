# Typed Polars models

`polars_list_math.typed_polars` builds typed Polars DataFrames from standard
slots dataclasses. Top-level rows are converted to tuples, while nested models
are converted to generated `NamedTuple` values for efficient Struct input.

```python
import polars_list_math.typed_polars as tp


@tp.model
class Item(tp.Model):
    value: str
    score: tp.F32


@tp.model
class Row(tp.Model):
    request_id: str = tp.field(polars_name="requestId")
    items: list[Item] = tp.field(default_factory=list)


rows = [Row(request_id="one", items=[Item("polars", 0.5)])]
frame = Row.to_frame_many(rows)
restored = list(Row.iter_frame(frame))
```

The `@tp.model` decorator applies `dataclass(slots=True)` automatically. An
explicit slots dataclass is also accepted when options such as `frozen=True`
or `kw_only=True` are needed.

## Names and dtypes

Use `polars_name=` for the physical Polars field name:

```python
request_id: str = tp.field(polars_name="requestId")
```

All physical names must be valid non-keyword Python identifiers and cannot
start with `_`, because nested Struct values use `NamedTuple` fields. Explicit
dtype aliases include `I8` through `I64`, unsigned integers, `F32`, `F64`,
timestamps, and durations. Unknown annotations fail immediately while the
model's module is imported; `tp.field(dtype=...)` can provide an intentional
storage dtype for an otherwise unsupported Python type.

## Nested and flat storage

Nested model annotations produce Struct values, and lists of models produce
ListStruct values. Use `flat=True` to expand either form into physical columns:

```python
@tp.model
class Metrics(tp.Model):
    views: int
    score: tp.F32


@tp.model
class Result(tp.Model):
    metrics: Metrics = tp.field(flat=True, flat_divider="__")
    history: list[Metrics] = tp.field(flat=True, flat_divider="__")
```

This produces `metrics__views`, `metrics__score`, `history__views`, and
`history__score`. Flat ListStruct fields are parallel list columns. Both forms
round-trip through `from_frame()` and support column expressions through
`Result.columns.metrics.fields.views` and `Result.columns.history.item.score`.

Typed dictionaries use `List[Struct[key, value]]` as their physical storage:

```python
weights: dict[str, float] = tp.field(default_factory=dict)
```

## Dictionaries and DataFrames

`to_dict()` and `from_dict()` use Python attribute names by default. Pass
`by_polars_name=True` to use physical names. Direct `from_dict()` rejects
unknown keys unless the model declares an `Extras` field.

DataFrame deserialization is permissive by default:

```python
row = Row.from_frame(frame)  # strict_schema=False
```

Unknown top-level columns, nested Struct/ListStruct fields, and flat-prefixed
columns are ignored when no `Extras` field exists. Declared `Extras` fields
capture unknown values at their own model level. Missing fields use dataclass
defaults; missing required fields raise `TypeError`.

Use `strict_schema=True` for an exact physical schema match, including column
order, dtypes, nested fields, and flat columns:

```python
row = Row.from_frame(frame, strict_schema=True)
```

Strict schema validation is unavailable for models with dynamic `Extras`.

See the runnable examples in `examples/typed_polars.py` and
`examples/typed_polars_flat.py`. More implementation details are documented in
`polars_list_math/typed_polars/README.md`.

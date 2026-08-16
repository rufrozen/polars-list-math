# Typed Polars

`polars_list_math.typed_polars` provides typed row schemas, nested column
references, reusable projections, and DataFrame conversion for Polars. It
requires Python 3.12 or newer and is included with `polars-list-math`.

## Import

Import the module with the `tp` alias:

```python
import polars_list_math.typed_polars as tp
```

This is the recommended style for application code, tests, and examples. It
keeps declarations readable, makes the origin of descriptors and dtype aliases
explicit, and avoids a long list of individual imports.

```python
class Result(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    score = tp.Field[tp.F32]()
```

## Schema fields

A `Schema` subclass describes both Python values and their physical Polars
representation:

```python
import polars_list_math.typed_polars as tp


class Result(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    score = tp.Field[float](default=0.0)
    tags = tp.Field[list[str]](default_factory=list)


row = Result(request_id="request-1", score=0.95)
```

The descriptor returns a Python value on an instance and a typed column
reference on the class:

```python
row.request_id                   # str
Result.request_id                # tp.Column[str]
Result.request_id.nested_expr()  # pl.col("requestId")
```

Python attribute names are used by constructors and `to_dict()`. The optional
`alias` is used at the Polars boundary:

```python
row.to_dict()
# {"request_id": "request-1", "score": 0.95, "tags": []}

row.to_dict(by_alias=True)
# {"requestId": "request-1", "score": 0.95, "tags": []}
```

## Dtypes

Common Python annotations map directly to Polars dtypes:

| Python annotation | Polars dtype |
| --- | --- |
| `str` | `pl.String` |
| `int` | `pl.Int64` |
| `float` | `pl.Float64` |
| `bool` | `pl.Boolean` |
| `bytes` | `pl.Binary` |
| `date` | `pl.Date` |
| `datetime` | `pl.Datetime("us")` |
| `timedelta` | `pl.Duration("us")` |
| `list[T]` | `pl.List(T)` |
| `dict[K, V]` | `pl.List(pl.Struct({"key": K, "value": V}))` |

Use the provided aliases when the physical width or time unit must be exact:

```python
class Event(tp.Schema):
    position = tp.Field[tp.I32]()
    score = tp.Field[tp.F32]()
    timestamp = tp.Field[tp.TimestampMs]()
    elapsed = tp.Field[tp.DurationUs]()
```

Available aliases are `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`,
`F32`, `F64`, `TimestampMs`, `TimestampUs`, `TimestampNs`, `DurationMs`,
`DurationUs`, and `DurationNs`. Prefix them with `tp.`. An explicit `dtype`
argument can override inference.

## Nested schemas

Use `Struct` for one nested schema and `ListStruct` for a list of nested
schemas:

```python
class Suggestion(tp.Schema):
    value = tp.Field[str]()
    score = tp.Field[tp.F32]()


class Completion(tp.Schema):
    prefix = tp.Field[str](alias="queryPrefix")
    suggestions = tp.ListStruct[Suggestion]()


class SearchRow(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    completion = tp.Struct[Completion](alias="completionData")
```

Nested class-level paths remain statically visible:

```python
SearchRow.completion.fields.prefix
# tp.Column[str]

SearchRow.completion.fields.suggestions.item.value
# tp.Column[str]
```

Use `nested_expr()` for nested Struct storage and `flat_expr()` for a column
produced by flat serialization.

## DataFrames

Schemas generate `pl.Schema` objects, serialize rows, and read rows back:

```python
row = SearchRow(
    request_id="request-1",
    completion=Completion(
        prefix="pol",
        suggestions=[Suggestion(value="polars", score=0.95)],
    ),
)

SearchRow.schema
frame = row.to_frame()
frame = SearchRow.to_frame_many([row])
restored = SearchRow.from_frame(frame)
rows = list(SearchRow.iter_frame(frame))
```

Pass `strict_schema=True` when reading if the complete DataFrame schema must
match exactly.

### Flat DataFrames

`to_flat_frame()` expands Struct fields into colon-separated scalar or list
columns:

```python
flat = row.to_flat_frame()
```

For the schema above, this produces columns such as
`completionData:queryPrefix` and `completionData:suggestions:value`. Set
`flat_alias` on `Struct` or `ListStruct` to customize a path segment only in
the flat representation:

```python
class SearchRow(tp.Schema):
    completion = tp.Struct[Completion](
        alias="completionData",
        flat_alias="completion",
    )
```

Use `to_flat_frame_many()` and `flat_schema()` for collections and schema-only
workflows.

## Views

A `View` declares a typed selection without creating another storage schema.
Each `ViewField` points to a source schema column:

```python
class SearchView(tp.View):
    request_id = tp.ViewField[str](SearchRow.request_id)
    prefix = tp.ViewField[str](SearchRow.completion.fields.prefix)
    values = tp.ViewField[list[str]](
        SearchRow.completion.fields.suggestions.item.value,
    )
    scores = tp.ViewField[list[float]](
        SearchRow.completion.fields.suggestions.item.score,
        alias="suggestion_scores",
    )
```

`select()` accepts eager or lazy frames. It detects nested or flat storage and
uses the matching expressions:

```python
nested_result = SearchView.select(frame)
flat_result = SearchView.select(flat)
lazy_result = SearchView.select(flat.lazy()).collect()
```

Missing sources become typed null columns without dropping rows. Read an eager
frame into typed objects with `from_frame()` or `iter_frame()`:

```python
views = SearchView.from_frame(frame)
views[0].request_id  # str
views[0].values      # list[str]
```

See the runnable [View example](../examples/typed_polars_view.py).

## Include schemas

`Include`, `IncludeStruct`, and `IncludeListStruct` derive a smaller storage
schema from shared source fields. They preserve aliases, dtypes, defaults, and
`default_factory` settings.

```python
class CompactSuggestion(tp.Schema):
    value = tp.Include(Suggestion.value)


class CompactCompletion(tp.Schema):
    prefix = tp.Include(Completion.prefix)
    suggestions = tp.IncludeListStruct(
        Completion.suggestions,
        CompactSuggestion,
    )


class CompactSearchRow(tp.Schema):
    request_id = tp.Include(SearchRow.request_id)
    completion = tp.IncludeStruct(
        SearchRow.completion,
        CompactCompletion,
    )
```

The result is a normal `Schema` that can be instantiated and converted to a
DataFrame independently:

```python
compact = CompactSearchRow(
    request_id="request-1",
    completion=CompactCompletion(
        prefix="pol",
        suggestions=[CompactSuggestion(value="polars")],
    ),
)

compact.to_frame()
```

Use `Include` with a nested column to expose that field as a top-level column:

```python
class PrefixOnly(tp.Schema):
    prefix = tp.Include(SearchRow.completion.fields.prefix)
```

See the runnable [Include example](../examples/typed_polars_include.py).

## Dynamic fields

Declare one `Extras` descriptor to retain fields not known when the class is
written:

```python
class FlexibleRow(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    extra = tp.Extras()


row = FlexibleRow(
    request_id="request-1",
    extra={"rank": 3, "debug": True},
)
```

Dynamic dtypes can be inferred from values or supplied with `extra_schema` to
`polars_schema()`, `to_frame()`, or `to_flat_frame()`.

## Static typing

Pyright and Pylance can check descriptor access, nested paths, view fields, and
dtype-aware annotations:

```python
row.request_id                           # str
SearchRow.request_id                     # tp.Column[str]
SearchRow.completion                     # tp.StructColumn[Completion]
SearchRow.completion.fields.prefix       # tp.Column[str]
SearchRow.completion.fields.suggestions  # tp.ListStructColumn[Suggestion]
```

Python's type system cannot synthesize an exact constructor signature from
runtime descriptor declarations. Constructor names and required fields are
validated at runtime, while field access and column paths remain statically
typed.

## Examples

Runnable examples are available in the repository:

- [typed schemas and DataFrames](../examples/typed_polars.py);
- [View projections](../examples/typed_polars_view.py);
- [Include schemas](../examples/typed_polars_include.py).

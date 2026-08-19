# typed_polars2

Experimental tuple-oriented typed Polars models. Public rows are standard
`dataclass(slots=True)` objects. During DataFrame construction top-level rows
become tuples and nested models become dynamically generated, cached-per-frame
`NamedTuple` values.

```python
import polars as pl
import polars_list_math.typed_polars2 as tp


@tp.model
class Item(tp.Model):
    value: str
    corrected_query: str = tp.field(polars_name="correctedQuery")


@tp.model
class Row(tp.Model):
    request_id: str = tp.field(polars_name="requestId")
    items: list[Item] = tp.field(default_factory=list)
    extra: tp.Extras = tp.extras(default_factory=dict)


row = Row(request_id="one", items=[Item("value", "corrected")])
frame = Row.to_frame_many([row])
```

Dictionary conversion uses Python attribute names by default. Pass
`by_polars_name=True` explicitly when reading or writing physical Polars names:

```python
data = row.to_dict(by_polars_name=True)
row = Row.from_dict(data, by_polars_name=True)
```

## DataFrame deserialization

`from_frame()` and `iter_frame()` use `strict_schema=False` by default. This
mode is intended for reading compatible DataFrames whose schemas are not
necessarily identical to the model schema.

| DataFrame difference | Without `Extras` | With `Extras` |
|---|---|---|
| Unknown top-level column | ignored | captured by top-level `Extras` |
| Unknown field inside Struct | ignored | captured by that nested model's `Extras` |
| Unknown field inside List[Struct] | ignored for every item | captured for every item |
| Unknown flat Struct column | ignored | captured by the flattened model |
| Unknown flat List[Struct] column | ignored | captured by every reconstructed item |
| Missing field with a dataclass default | default is used | default is used |
| Missing required field | `TypeError` | `TypeError` |

`Extras` only applies at the level where it is declared. A top-level `Extras`
field does not capture unknown fields from a nested Struct; the nested model
must declare its own `Extras` field to preserve them.

For example, both the unknown top-level column and the unknown nested field are
ignored here:

```python
frame = pl.DataFrame(
    {
        "requestId": ["one"],
        "items": [[{"value": "result", "unknownNested": 1}]],
        "unknownTop": [True],
    }
)

row = Row.from_frame(frame)  # strict_schema=False
```

This permissive handling is specific to DataFrame deserialization. Direct
`Row.from_dict(...)` calls continue to reject unknown keys unless the model
declares `Extras`.

Pass `strict_schema=True` to require an exact schema, including physical flat
columns and nested Struct fields:

```python
row = Row.from_frame(frame, strict_schema=True)
```

Exact schema validation is unavailable for models containing dynamic `Extras`.
Strict validation compares the complete physical schema, including column
order, dtypes, nested Struct fields, and flat Struct/ListStruct columns.

All physical names must be valid non-keyword Python identifiers and cannot
start with `_`. This guarantees that every nested Struct can be represented by
a NamedTuple; this implementation deliberately has no dictionary fallback.

Flat structs and lists of structs use `tp.field(flat=True)`. Dynamic extras are
collected across the input rows before the frame-specific physical tuple plan
is compiled.

Typed dictionaries are stored without a dictionary fallback in the frame
construction path: `dict[K, V]` becomes `List[Struct[key: K, value: V]]`, and
each physical key/value item is a generated named tuple. See the complete
[`examples/typed_polars2.py`](../../examples/typed_polars2.py) example.

The `@tp.model` decorator resolves every annotation immediately. An unknown
scalar type, including one nested inside `list` or `dict`, raises `TypeError`
while the module containing the model is imported. Use `tp.field(dtype=...)`
when an otherwise unsupported Python type has an intentional Polars storage
type.

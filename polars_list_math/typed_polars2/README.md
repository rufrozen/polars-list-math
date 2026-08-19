# typed_polars2

Experimental tuple-oriented typed Polars models. Public rows are standard
`dataclass(slots=True)` objects. During DataFrame construction top-level rows
become tuples and nested models become dynamically generated, cached-per-frame
`NamedTuple` values.

```python
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

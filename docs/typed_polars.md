# Typed Polars schemas

`polars_list_math.typed_polars` declares typed row models, their fixed Polars
storage schema, and typed expressions for projections.

```python
import polars_list_math.typed_polars as tp

class Item(tp.Schema):
    value = tp.Field[str]()
    score = tp.Field[tp.F32]()

class Payload(tp.Schema):
    title = tp.Field[str]()
    items = tp.ListStruct[Item](flat=True)

class Row(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    payload = tp.Struct[Payload](flat=False)
```

Fields are descriptors: `row.request_id` is a `str`, while
`Row.request_id` is a typed `Column[str]`. Nested references remain typed:

```python
Row.payload.fields.title.expr()
Row.payload.fields.items.item.score.expr()
```

## One fixed, hybrid representation

Every `Struct` and `ListStruct` explicitly chooses its physical storage:

- `flat=False` (the default) stores a Polars `Struct` or `List(Struct)` column;
- `flat=True` expands that node into separate columns;
- `flat_divider=":"` controls the separator at that particular boundary.

The choice is local, so nested declarations form a hybrid schema:

```python
class Details(tp.Schema):
    source = tp.Field[str]()

class Item(tp.Schema):
    value = tp.Field[str]()
    details = tp.Struct[Details]()  # remains Struct

class Row(tp.Schema):
    items = tp.ListStruct[Item](
        alias="matches",
        flat=True,
        flat_divider=".",
    )
```

This produces `matches.value: List(String)` and
`matches.details: List(Struct({source: String}))`. A nested flat node may use a
different divider; each divider applies only to its own boundary.

There is no second nested/flat representation. `schema`, `polars_schema()`,
`to_frame()`, `to_frame_many()`, `from_frame()`, `iter_frame()`, and column
`expr()` all use the same declared layout.

## Rows and DataFrames

```python
row = Row(
    request_id="one",
    payload=Payload(title="Result", items=[]),
)

frame = row.to_frame()
assert frame.schema == Row.schema
restored = Row.from_frame(frame, strict_schema=True)
```

Use `to_dict()` and `from_dict()` for the logical Python representation.
`by_alias=True` switches dictionary keys to Polars aliases. Unlike DataFrame
storage, dictionaries keep nested model structure and do not flatten fields.

Supported annotations include scalar Python types, nullable unions, lists,
dictionaries, nested schemas, and the explicit integer, float, timestamp, and
duration aliases exported by the module.

## Views

Views select typed fields from the schema's fixed representation:

```python
class RowView(tp.View):
    identifier = tp.ViewField[str](Row.request_id)
    title = tp.ViewField[str](Row.payload.fields.title)

selected = RowView.select(frame)
objects = RowView.from_frame(frame)
```

`select()` supports eager and lazy frames. A missing physical source becomes a
typed null column; it does not probe for an alternative representation.

## Includes and dynamic fields

`Include`, `IncludeStruct`, and `IncludeListStruct` copy fields into partial
schemas. Struct includes preserve `flat` and `flat_divider` from their source.

Declare one `Extras()` descriptor to capture undeclared values. When building a
frame, their dtypes are inferred across rows or supplied with `extra_schema`.
For flattened Structs, nested extras are expanded using that Struct's divider.

See the runnable examples in `examples/typed_polars.py`,
`examples/typed_polars_view.py`, and `examples/typed_polars_include.py`.

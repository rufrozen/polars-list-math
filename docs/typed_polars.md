# Typed Polars models

`polars_list_math.typed_polars` separates Python data models from Polars storage
metadata. Models contain only ordinary dataclass annotations and defaults; a
`Schema` is the single source of physical column names and Polars dtypes.

```python
from dataclasses import dataclass, field

import polars_list_math.typed_polars as tp


class ItemSchema(tp.Schema):
    value = tp.Column[str]()
    score = tp.Column[tp.F32]()


class RowSchema(tp.Schema):
    request_id = tp.Column[str](polars_name="requestId")
    items = tp.ListStruct[ItemSchema]()


@dataclass
class Item:
    value: str
    score: float


@tp.model(schema=RowSchema)
@dataclass
class Row:
    request_id: str
    items: list[Item] = field(default_factory=list)


row = Row("one", [Item("polars", 0.5)])
frame = RowSchema.to_frame(row)
assert RowSchema.from_frame(Row, frame) == row
```

Nested models are ordinary dataclasses and do not need a library base class or
decorator. The root dataclass must use `@tp.model(schema=...)`; this recursively
validates the model and builds the cached physical serialization plan. Defaults
and factories are declared with `dataclasses.field`.

The decorator creates a `tp.Builder` for the model/schema pair and stores it on
the root model. It owns the recursive validation result and cached physical
plan. `tp.Builder.for_model(Row, schema=RowSchema)` returns that instance.
`Schema` remains stateless and only uses the builder attached to an explicitly
supplied model (or to the row passed to `to_frame`).

By default, model and schema fields are matched by intersection. Schema-only
fields are ignored by model conversion. Model-only fields are also excluded
from the DataFrame and must declare a dataclass `default` or `default_factory`
so deserialization can construct the model. Matching fields still require a
compatible Python type and scalar/Struct/ListStruct shape.

Use `@tp.model(schema=RowSchema, strict=True)` to require exact field sets at
every nested level. This structural `strict` option is independent of the
Polars value-validation option passed to `Schema.to_frame*`.

DataFrame conversion lives on the schema: use `RowSchema.to_frame(row)` and
`RowSchema.to_frame_many(Row, rows)` for serialization. Deserialization also
takes the model type explicitly: `RowSchema.from_frame(Row, frame)` and
`RowSchema.iter_frame(Row, frame)`. The schema validates that rows and explicit
model arguments exactly match the registered root type.

`RowSchema.polars_schema()` returns the complete schema declaration, while
`RowSchema.model_schema(Row)` returns the cached physical schema containing
only the fields selected by that model.

`to_dict()` and `from_dict()` use Python names by default. Pass
`by_polars_name=True` to use schema names. `from_frame()` and `iter_frame()` are
permissive by default and ignore columns not selected by the model; pass
`strict_schema=True` to require the model's exact physical schema.

Schema columns also provide typed expressions. Use `.fields` after `Struct` and
`.item` after `ListStruct`, for example
`RowSchema.items.item.score.expr()`.

## Runtime dictionary columns

`FlatDict[T]` expands a model field annotated as `dict[str, T]` into physical
columns selected at runtime. Bind the complete ordered key set in a `Context`
when building a schema or DataFrame:

```python
class MetricsSchema(tp.Schema):
    request_id = tp.Column[str](polars_name="requestId")
    metrics = tp.FlatDict[float]()


@tp.model(schema=MetricsSchema)
class MetricsRow:
    request_id: str
    metrics: dict[str, float] = field(default_factory=dict)


context = tp.Context().bind(MetricsSchema.metrics, ["views", "conversion"])
row = MetricsRow("one", {"views": 10.0})
frame = MetricsSchema.to_frame(row, context=context)
```

The frame contains `metrics_views` and `metrics_conversion`; a missing key is
stored as null. `polars_name` controls the prefix and `divider` defaults to
`_`. With no context, or no binding for a particular `FlatDict`, its key set
is empty. A non-empty runtime dictionary containing unbound keys is rejected to
prevent silent data loss.

Reverse conversion does not take a context. `from_frame()` and `iter_frame()`
discover dynamic keys by prefix and omit null values when rebuilding the
dictionary. `FlatDict.key_expr("views")` creates an expression for one dynamic
physical column. Physical plans are cached by the context binding snapshot.

For a nested field, bind its path from the root schema, for example
`context.bind(RowSchema.payload.fields.metrics, keys)`. Binding the original
nested declaration instead applies the same keys as a fallback everywhere that
nested schema is reused; path bindings can override it independently.

## Flat storage

Flattening is a storage property of the schema, not the dataclass model.
`FlatStruct` and `FlatListStruct` expand nested values into sibling physical
columns; `divider` controls the physical column separator:

```python
class FlatRowSchema(tp.Schema):
    item = tp.FlatStruct[ItemSchema](divider="__")
    history = tp.FlatListStruct[ItemSchema]()
```

This produces scalar columns such as `item__value` and, with the default `_`
divider, parallel list columns such as `history_value`. Schema expressions
point to those physical columns, and model DataFrame conversion transparently
flattens and reconstructs the nested dataclass values.

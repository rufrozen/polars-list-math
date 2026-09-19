# Null schema discovery and cleaning

Two Python modules provide three top-level functions:

- `find_null_schema(lf) -> pl.Schema` inspects a LazyFrame and describes the
  columns and nested fields without any non-null leaf values.
- `clean_schema(schema, subschema) -> pl.Schema` subtracts any compatible
  subschema from an existing schema without reading data. The result of
  `find_null_schema` is one possible subschema.
- `apply_schema(lf, schema) -> pl.LazyFrame` selects and casts the frame to the
  resulting schema, without collecting it.

```python
import polars as pl
from polars_list_math import apply_schema, clean_schema, find_null_schema

lf = pl.LazyFrame(
    {
        "id": [1, 2],
        "items": [
            [{"a": 10, "b": None, "c": "hello"}],
            [{"a": None, "b": None, "c": None}],
        ],
        "empty": [[[None]], None],
    },
    schema={
        "id": pl.Int64,
        "items": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String, "c": pl.String})),
        "empty": pl.List(pl.List(pl.Null)),
    },
)

null_schema = find_null_schema(lf)
assert null_schema == pl.Schema({
    "items": pl.List(pl.Struct({"b": pl.Null})),
    "empty": pl.List(pl.List(pl.Null)),
})

schema = clean_schema(lf.collect_schema(), null_schema)
assert schema == pl.Schema({
    "id": pl.Int64,
    "items": pl.List(pl.Struct({"a": pl.Int64, "c": pl.String})),
})

cleaned = apply_schema(lf, schema)
result = cleaned.collect()
```

## What counts as absent

Discovery examines actual values across all rows, rather than only checking
the declared dtype. A typed `Int64` or `String` field containing only nulls
becomes `Null` in the returned mask. A field with even one non-null value is
omitted. Zero, `False`, empty strings, and floating-point NaN are values.

Nesting can combine `List`, `Struct`, and fixed-size `Array` at any depth.
Containers keep their structure in the mask; array dimensions are preserved.
For example, a wholly absent `List(List(Struct({a: Int64, b: String})))` becomes
`List(List(Struct({a: Null, b: Null})))`. Cleaning removes both fields, then
their now-empty struct, then its list wrappers, and finally the whole column.

Null containers, empty lists, and empty structs contribute no leaf values.
For an empty frame every declared leaf is absent. If any element elsewhere
has a value, its leaf survives. Discovery respects filters already in the
LazyFrame's plan.

## Execution and schema masks

Discovery executes one aggregate query and returns one boolean per typed
leaf to Python, without collecting the full frame. Known `Null` dtypes and
empty structs need no scan. There is no Python callback per row.

Applying the cleaned schema executes the source again when the resulting
LazyFrame is collected. Use a replayable source: a one-shot iterator passed to
`dfs_to_lazy_df` is consumed by discovery. The data should remain unchanged
between discovery and applying its result.

`clean_schema` is independent of null discovery: a subschema can describe any
columns and fields to remove, including populated fields. Container types,
array sizes, and scalar dtypes must match the source schema. Structs may
include only a subset of fields. `Null` is also accepted at any position as a
marker removing that entire subtree, so discovery results remain valid inputs.
Unknown names or incompatible subschemas raise `ValueError`. Cleaning does not
read data or check values.

For example, remove the populated `a` field from the frame above:

```python
subschema = pl.Schema({"items": pl.List(pl.Struct({"a": pl.Int64}))})
schema = clean_schema(lf.collect_schema(), subschema=subschema)
cleaned = apply_schema(lf, schema)
# items now contains b: String and c: String; id and empty are retained.
```

Retained columns and fields keep their order and dtypes. `apply_schema` uses
strict Polars casts and preserves parent nulls, null list elements, empty
lists, and row order. It also supports reordering columns and casting their
dtypes when a target schema is supplied directly. Missing top-level columns
raise a Polars error; missing nested fields follow Polars cast semantics and
are filled with nulls. An empty target follows Polars empty-projection
behavior and produces a frame with zero columns and zero rows.

Polars references: [lazy schemas](https://docs.pola.rs/user-guide/lazy/schemas/)
and [expression casts](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.cast.html).

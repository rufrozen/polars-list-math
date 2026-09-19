"""Discover absent nested fields, clean a schema, and apply it lazily."""

import polars as pl
from polars_list_math import apply_schema, clean_schema, find_null_schema

lf = pl.LazyFrame(
    {
        "id": [1, 2],
        "items": [[{"a": 10, "b": None, "c": "hello"}], None],
        "empty": [[[None]], None],
    },
    schema={
        "id": pl.Int64,
        "items": pl.List(pl.Struct({"a": pl.Int64, "b": pl.String, "c": pl.String})),
        "empty": pl.List(pl.List(pl.Null)),
    },
)

null_schema = find_null_schema(lf)
schema = clean_schema(lf.collect_schema(), null_schema)
cleaned = apply_schema(lf, schema)

print("Null schema:", null_schema)
print("Cleaned schema:", schema)
print(cleaned.collect())

# A regular typed subschema can also remove populated fields.
subschema = pl.Schema({"items": pl.List(pl.Struct({"a": pl.Int64}))})
without_a = clean_schema(lf.collect_schema(), subschema=subschema)
print(apply_schema(lf, without_a).collect())

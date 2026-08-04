# JSON object items

`json_object_items` converts each JSON object string into a list of `key`/`value`
structs:

```python
import polars as pl
import polars_list_math  # registers the expression namespaces

df = pl.DataFrame({"json": ['{"name":"Ada","active":true,"note":null}']})

result = df.select(pl.col("json").str.json_object_items().alias("items"))
```

The output dtype is:

```python
pl.List(pl.Struct({"key": pl.String, "value": pl.String}))
```

String values are preserved. Null stays null, while numbers, booleans, arrays,
and objects are encoded as compact JSON strings. Valid JSON values that are not
objects return null. Invalid JSON also returns null by default; pass `strict=True`
to raise a `ComputeError` instead.

# JSON array values

`json_array_values` converts each JSON array string into a list of strings:

```python
import polars as pl
import polars_list_math  # registers the expression namespaces

df = pl.DataFrame({"json": ['["Ada",true,null,{"score":10}]']})

result = df.select(pl.col("json").str.json_array_values().alias("values"))
```

The output dtype is `pl.List(pl.String)`. String values are preserved and null
stays null, while numbers, booleans, arrays, and objects are encoded as compact
JSON strings. Valid JSON values that are not arrays return null. Invalid JSON
also returns null by default; pass `strict=True` to raise a `ComputeError`
instead.

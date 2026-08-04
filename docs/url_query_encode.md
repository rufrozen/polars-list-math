# URL query encoding

`url_query_encode` is an expression-oriented equivalent of
`urllib.parse.urlencode`:

```python
import polars as pl
from polars_list_math import url_query_encode

df = pl.DataFrame({"name": ["Ada Lovelace"], "page": [2]})
result = df.select(url_query_encode(pl.struct("name", "page")).alias("query"))
```

The input may be a Struct expression or a `List(Struct)` expression with `key`
and `value` fields. The latter accepts the output of `str.json_object_items()`
and preserves repeated keys. Pass `doseq=True` to encode list values as repeated
query parameters.

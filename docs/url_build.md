# URL building

`url_build` is an expression-oriented equivalent of `urllib.parse.urlunparse`:

```python
import polars as pl
from polars_list_math import url_build

result = df.select(
    url_build(
        scheme=pl.lit("https"),
        netloc=pl.col("host"),
        path=pl.col("path"),
        query=pl.col("query"),
    ).alias("url")
)
```

All six components (`scheme`, `netloc`, `path`, `params`, `query`, and
`fragment`) are optional expressions. Missing and null components are treated as
empty strings. Components are assembled without percent encoding.

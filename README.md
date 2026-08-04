# polars-list-math

`polars-list-math` is a Python/Rust package with list, JSON, and URL expression
helpers for Polars.

Import the package once to register extra methods on `Expr.list` and `Expr.str`:

```python
import polars as pl
import polars_list_math  # noqa: F401
```

## Install

```bash
pip install polars-list-math
# or
uv add polars-list-math
```

Requires Python 3.12+ and `polars>=1.39.3`.

## Methods

| Method | Result | Docs |
| --- | --- | --- |
| `Expr.list.zip(...)` | Zip lists into `list[struct]` | [Polars proposal](https://github.com/pola-rs/polars/issues/22719) |
| `Expr.list.combinations(...)` | Pair items in one list, optionally skipping self-pairs | [Combinations](https://github.com/rufrozen/polars-list-math/blob/main/docs/combinations.md) |
| `Expr.list.combinations_to(...)` | Pair each item with each item from another list | [Combinations](https://github.com/rufrozen/polars-list-math/blob/main/docs/combinations.md) |
| `Expr.list.similarity(...)` | Weighted similarity between two lists | [Similarity](https://github.com/rufrozen/polars-list-math/blob/main/docs/similarity.md) |
| `Expr.list.mean_similarity(...)` | Mean similarity inside a nested-list row | [Similarity](https://github.com/rufrozen/polars-list-math/blob/main/docs/similarity.md) |
| `Expr.list.mean_similarity_to(...)` | Mean similarity to reference nested lists | [Similarity](https://github.com/rufrozen/polars-list-math/blob/main/docs/similarity.md) |
| `Expr.str.json_object_items(...)` | Convert a JSON object to `list[struct[key, value]]` | [JSON object items](https://github.com/rufrozen/polars-list-math/blob/main/docs/json_object_items.md) |
| `Expr.str.json_array_values(...)` | Convert a JSON array to `list[str]` | [JSON array values](https://github.com/rufrozen/polars-list-math/blob/main/docs/json_array_values.md) |

Top-level expression helpers:

| Function | Result | Docs |
| --- | --- | --- |
| `url_query_encode(expr, doseq=False)` | Encode a Struct or key/value list as a URL query string | [URL query encoding](https://github.com/rufrozen/polars-list-math/blob/main/docs/url_query_encode.md) |
| `url_build(...)` | Assemble a URL from optional component expressions | [URL building](https://github.com/rufrozen/polars-list-math/blob/main/docs/url_build.md) |

The package also exports `json_object_items(expr)` and
`json_array_values(expr)` as top-level equivalents of the `Expr.str` methods.
The Python helper `py_list_similarity(...)` computes weighted similarity for
plain Python sequences.

If a future Polars release ships native methods with the same names, this
package leaves the native implementation untouched.

## Quick Examples

```python
df = pl.DataFrame(
    {
        "a": [[1, 2, 3]],
        "b": [[2, 1, 3]],
        "groups": [[[1, 2, 3], [1, 2, 3], [4, 5, 6]]],
    }
)

df.with_columns(
    pl.col("a").list.similarity("b").alias("similarity"),
    pl.col("groups").list.mean_similarity().alias("mean_similarity"),
    pl.col("a").list.combinations().alias("pairs"),
    pl.col("a").list.zip("b", fields=["a", "b"]).alias("zipped"),
)
```

Convert JSON strings to nested Polars values. String values stay strings, JSON
null stays null, and other values become compact JSON strings. Invalid JSON
returns null by default; pass `strict=True` to raise instead.

```python
json_df = pl.DataFrame(
    {
        "object": ['{"name":"Ada","active":true,"note":null}'],
        "array": ['["python",42,true,null]'],
    }
)

json_df.select(
    pl.col("object").str.json_object_items().alias("items"),
    pl.col("array").str.json_array_values().alias("values"),
)
```

Encode query parameters and build URLs from expressions:

```python
from polars_list_math import url_build, url_query_encode

pages = pl.DataFrame(
    {
        "host": ["example.com"],
        "path": ["/search"],
        "term": ["rust polars"],
        "page": [2],
    }
)

query = url_query_encode(pl.struct(q=pl.col("term"), page=pl.col("page")))

pages.select(
    url_build(
        scheme=pl.lit("https"),
        netloc="host",
        path="path",
        query=query,
    ).alias("url")
)
```

More complete, runnable examples are available in the
[`examples`](https://github.com/rufrozen/polars-list-math/tree/main/examples)
directory.

## Development

```bash
make install
make develop
make test
```

Build and check the package locally:

```bash
make check-dist
```

Release steps are in
[docs/publishing.md](https://github.com/rufrozen/polars-list-math/blob/main/docs/publishing.md).

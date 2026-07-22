# polars-list-math

`polars-list-math` is a Python/Rust package with list-oriented expression
helpers for Polars.

Import the package once to register extra methods on `Expr.list`:

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
| `Expr.list.combinations(...)` | Pair each item with itself and later items in the same list | [docs/combinations.md](docs/combinations.md) |
| `Expr.list.combinations_to(...)` | Pair each item with each item from another list | [docs/combinations.md](docs/combinations.md) |
| `Expr.list.similarity(...)` | Weighted similarity between two lists | [docs/similarity.md](docs/similarity.md) |
| `Expr.list.mean_similarity(...)` | Mean similarity inside a nested-list row | [docs/similarity.md](docs/similarity.md) |
| `Expr.list.mean_similarity_to(...)` | Mean similarity to reference nested lists | [docs/similarity.md](docs/similarity.md) |

The Python helper `py_list_similarity(...)` computes the same weighted
similarity for plain Python sequences.

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

## Development

```bash
make install
make develop
make test
```

Build and upload:

```bash
make build
make publish
```

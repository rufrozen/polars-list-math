# Similarity

`polars-list-math` adds weighted list-similarity helpers to Polars'
`Expr.list` namespace.

Use these when list order matters. A match near the beginning of a list is worth
more than the same match near the end.

## API

```python
import polars as pl
import polars_list_math  # registers Expr.list helpers
from polars_list_math import py_list_similarity
```

```python
pl.col("a").list.similarity("b", p=0.9)
pl.col("lists").list.mean_similarity(p=0.9)
pl.col("targets").list.mean_similarity_to("references", p=0.9)

py_list_similarity([1, 2, 3], [2, 1, 3], p=0.9)
```

`similarity` compares two list columns row by row and returns `Float64`.

`mean_similarity` expects a nested list column, such as
`List(List(Int64))` or `List(List(String))`. For each inner list, it returns the
mean similarity against the other inner lists in the same row.

`mean_similarity_to` also expects nested lists. It compares each target inner
list against all reference inner lists in the same row.

## Formula

Each item receives a positional weight:

$$
w(rank) = p^{rank - 1}
$$

Ranks start at `1`. With the default `p = 0.9`, the first few weights are:

| Rank | Weight |
| ---: | ---: |
| 1 | 1.0000 |
| 2 | 0.9000 |
| 3 | 0.8100 |
| 4 | 0.7290 |

For each item `d`, define its weight in list `A`:

$$
W_A(d) =
\begin{cases}
w(rank_A(d)), & d \in A \\
0, & d \notin A
\end{cases}
$$

The similarity is weighted Jaccard similarity:

$$
similarity(A, B) =
\frac{\sum_{d \in A \cup B} \min(W_A(d), W_B(d))}
{\sum_{d \in A \cup B} \max(W_A(d), W_B(d))}
$$

The result is always between `0.0` and `1.0`.

## Examples

```python
df = pl.DataFrame(
    {
        "a": [[1, 2, 3], [1, 2, 3], []],
        "b": [[1, 2, 3], [2, 1, 3], []],
    }
)

df.with_columns(
    pl.col("a").list.similarity("b").alias("similarity")
)
```

```text
shape: (3, 3)
┌───────────┬───────────┬────────────┐
│ a         ┆ b         ┆ similarity │
│ ---       ┆ ---       ┆ ---        │
│ list[i64] ┆ list[i64] ┆ f64        │
╞═══════════╪═══════════╪════════════╡
│ [1, 2, 3] ┆ [1, 2, 3] ┆ 1.0        │
│ [1, 2, 3] ┆ [2, 1, 3] ┆ 0.928826   │
│ []        ┆ []        ┆ 1.0        │
└───────────┴───────────┴────────────┘
```

Mean similarity within one row:

```python
df = pl.DataFrame(
    {"lists": [[[1, 2, 3], [1, 2, 3], [4, 5, 6]]]},
    schema={"lists": pl.List(pl.List(pl.Int64))},
)

df.with_columns(
    pl.col("lists").list.mean_similarity().alias("mean_similarity")
)
```

Mean similarity to a reference nested-list column:

```python
df.with_columns(
    pl.col("target_lists")
    .list.mean_similarity_to("reference_lists")
    .alias("mean_similarity")
)
```

## Nulls And Types

`similarity` returns null when either input list is null.

Null items inside a list are ignored. If both lists are empty, or only contain
null items, similarity is `1.0`.

The native implementation supports primitive list inner dtypes: booleans,
integers, unsigned integers, floats, strings, binary values, and temporal values.
`Null` lists are also accepted and behave like lists with no document IDs.

When two non-null list dtypes are compared, their inner dtypes must match
exactly. Cast explicitly first when you want `Int32` and `Int64`, or any other
different dtypes, to be treated as the same ID space.

Duplicates use the best position: only the first occurrence of an item
contributes its weight.

# Kendall tau

These helpers compare **ranked ID lists**, like `similarity`: an ID's rank is
its position in the list. They compute Kendall's tau on the positions of the
distinct IDs shared by both lists. Values of the IDs themselves are not scores.
For example, `["a", "b", "c"]` and `["c", "b", "a"]` have tau `-1.0`.

## API

```python
import polars as pl
import polars_list_math
from polars_list_math import (
    list_kendall_tau,
    list_mean_kendall_tau,
    list_mean_kendall_tau_to,
    py_list_kendall_tau,
)

list_kendall_tau("a", "b", normalize=False)
list_mean_kendall_tau("lists", normalize=False)
list_mean_kendall_tau_to("targets", "references", normalize=False)
py_list_kendall_tau([1, 2, 3], [2, 1, 3], normalize=False)  # 1/3

# Equivalent runtime namespace methods:
pl.col("a").list.kendall_tau("b", normalize=True)
pl.col("lists").list.mean_kendall_tau(normalize=True)
pl.col("targets").list.mean_kendall_tau_to("references", normalize=True)
```

Expression inputs are column names or Polars expressions. `list_kendall_tau`
returns `Float64`. The two mean helpers return `List(Float64)`, with one score
per target inner list. `normalize` is a keyword-only boolean and defaults to
`False`; Kendall tau has no positional decay parameter `p`.

The self mean excludes comparison with the same list position, while distinct
positions containing identical lists still count as peers. Reference means
include every defined target/reference comparison.

## Formula and normalization

After keeping only common IDs and their first occurrence, let `C` be the number
of pairs ordered the same way in both lists and `D` the number ordered in
opposite ways. For `n >= 2` common IDs:

```text
tau = (C - D) / (n * (n - 1) / 2)
normalized_tau = (tau + 1) / 2
```

Ranks are unique after deduplication, so tau-a and tau-b coincide. See the
[SciPy definition of Kendall tau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)
for the general formula including tied ranks. To reproduce this API using
SciPy, pass the positions of the common IDs in each list, rather than the ID
lists themselves.

| Ranking agreement | `normalize=False` | `normalize=True` |
| --- | ---: | ---: |
| Same order | 1 | 1 |
| Equal concordant and discordant counts | 0 | 0.5 |
| Reversed order | -1 | 0 |

IDs present in only one list are ignored. Consequently, two lists with little
overlap can still have tau `1.0`: use `similarity` or `containment` as well when
overlap matters.

## Nulls, duplicates, and types

- Null lists produce null. Null IDs within a list are ignored.
- Duplicate IDs use their first occurrence.
- Fewer than two common distinct IDs produce null (`None` in Python), including
  empty, singleton, and disjoint lists. Normalization preserves null.
- Means exclude null and undefined comparisons. A target with no defined
  comparisons gets null. A null outer target stays null; an empty outer target
  produces `[]`. Null or empty reference lists give null for every target.
- Expressions support `List` and fixed-size `Array` at either nesting level,
  including scalar list literals that broadcast across rows.
- Supported IDs are booleans, integers, floats, strings, binary and temporal
  values. Non-null inner dtypes must match exactly; cast explicitly when needed.
  `List(Null)` is compatible with any supported ID dtype.
- ID equality follows `similarity`, including bitwise floating-point identity
  (`-0.0` and `+0.0` are different IDs). The Python API rejects mixed primitive
  types and nested IDs.

## Example

```python
frame = pl.DataFrame(
    {
        "a": [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1]],
        "b": [[1, 2, 3], [2, 1, 3], [3, 2, 1], [1]],
    }
)

frame.select(
    list_kendall_tau("a", "b").alias("tau"),
    list_kendall_tau("a", "b", normalize=True).alias("normalized"),
)
# tau:        [1.0, 0.333333..., -1.0, null]
# normalized: [1.0, 0.666666...,  0.0, null]
```

The Rust plugin runs automatically when the native extension is built. A Python
fallback provides the same results without a compiled library. Both count
discordant pairs with a Fenwick tree instead of enumerating every pair. For
input lengths `n` and `m`, a comparison takes `O((n + m) log(n + m))` time and
`O(n + m)` memory. Nested means prepare each inner list's ranks once per row,
and self means score each unordered pair once.

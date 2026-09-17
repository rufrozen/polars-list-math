# SERP containment

Containment measures the fraction of one search result list found in another:

```text
containment(A, B) = |A ∩ B| / |A|
```

`A` and `B` are sets of unique non-null document IDs. Order and duplicates do
not affect the score. An empty `A` scores `0.0`, including when both sets are
empty. Scores lie between `0.0` and `1.0`.

This is a directed metric: `[1, 2]` against `[1, 2, 3]` scores `1.0`, whereas
the reverse scores `2/3`. Use [similarity](similarity.md) when ranks should
contribute positional weights.

## API

```python
import polars as pl
import polars_list_math  # registers the namespace methods
from polars_list_math import (
    list_containment,
    list_mean_containment,
    list_mean_containment_to,
    py_list_containment,
)

list_containment("a", "b")
list_mean_containment("groups")
list_mean_containment_to("targets", "references")
py_list_containment([1, 2], [1, 2, 3])  # 1.0

# Equivalent runtime namespace methods:
pl.col("a").list.containment("b")
pl.col("groups").list.mean_containment()
pl.col("targets").list.mean_containment_to("references")
```

The top-level expression helpers support static type checking. All three use
native Rust plugins when the extension is built, with Python fallbacks otherwise.
There is no positional-weight parameter.

| Helper | Input | Output |
| --- | --- | --- |
| `list_containment` | Two `List(primitive)` columns | `Float64` per row |
| `list_mean_containment` | One `List(List(primitive))` column | `List(Float64)` per row |
| `list_mean_containment_to` | Target and reference nested-list columns | `List(Float64)` per row |
| `py_list_containment` | Two Python sequences | Python `float` |

`mean_containment` returns one score per inner list: the mean of
`containment(target, peer)` over all other non-null inner lists. Self pairs
are excluded by index; separate lists with equal contents still count as peers.
Each pair contributes a separately normalized score in each direction.
For `[[1, 2], [1, 2, 3], [4]]`, the output is `[0.5, 1/3, 0.0]`.

`mean_containment_to` averages `containment(target, reference)` for each target
over the non-null reference lists. The target is always the denominator.
All references count, including equal lists and self matches when the same
column is supplied twice. For targets `[[1, 2], [1, 2, 3]]` and references
`[[1, 2], [2]]`, the output is `[0.75, 0.5]`.

## Nulls and types

- A null input list produces a null pairwise score.
- Null document IDs are ignored; an all-null list is an empty set.
- A null target group produces a null output row.
- A null inner target produces a null score; null peers and references are skipped.
- With no valid peers or references, each target score is null. An empty target
  group produces `[]`. A singleton self group produces `[null]`.
- Empty inner lists are valid comparisons and score zero in either direction.
- Native ID dtypes follow similarity: booleans, integers, floats, strings,
  binary and temporal values. Compared non-Null inner dtypes must match exactly;
  cast explicitly when needed. Struct and nested document IDs are unsupported.
- The Python helper accepts homogeneous primitive scalar IDs and rejects mixed
  scalar types. Float IDs follow similarity's bitwise identity rules (including
  signed zero and NaN); use string URL IDs for typical SERP comparisons.

See the runnable [example](../examples/containment.py) and
[native performance experiment](../performance/native_plugins/list_containment).

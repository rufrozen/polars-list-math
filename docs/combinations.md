# Combinations

`polars-list-math` adds pair-building helpers to Polars' `Expr.list` namespace.
They are useful when you need to compare, score, or explode pairs from list
columns.

Both functions accept only list expressions and return list expressions. The
result dtype is always `List(Struct(...))`.

## API

```python
import polars as pl
import polars_list_math  # registers Expr.list helpers
```

```python
pl.col("a").list.combinations(
    left_value="left_value",
    right_value="right_value",
    with_index=False,
    left_index=None,
    right_index=None,
    skip_null=False,
)
```

`combinations` builds all pairs inside one list where `left_index <= right_index`.

```python
pl.col("a").list.combinations_to(
    "b",
    left_value="left_value",
    right_value="right_value",
    with_index=False,
    left_index=None,
    right_index=None,
    skip_null=False,
)
```

`combinations_to` builds all pairs from the left list to the target list in the
same row.

## Output Fields

By default, each pair is a struct with two fields:

| Field | Meaning |
| --- | --- |
| `left_value` | Value from the left/input list |
| `right_value` | Value from the right/target list |

Set `with_index=True` to include `left_index` and `right_index`. Passing
`left_index` or `right_index` also includes index fields, even when
`with_index=False`. If only one index name is provided, the other uses its
default name.

The emitted field names can be changed with keyword arguments.

## Examples

All pairs inside one list:

```python
df = pl.DataFrame({"a": [[10, 20, 30]]})

df.select(
    pl.col("a").list.combinations().alias("pairs")
)
```

```text
[
  [
    {"left_value": 10, "right_value": 10},
    {"left_value": 10, "right_value": 20},
    {"left_value": 10, "right_value": 30},
    {"left_value": 20, "right_value": 20},
    {"left_value": 20, "right_value": 30},
    {"left_value": 30, "right_value": 30},
  ]
]
```

All pairs from one list to another:

```python
df = pl.DataFrame(
    {
        "left": [[1, 2]],
        "right": [["a", "b"]],
    }
)

df.select(
    pl.col("left").list.combinations_to("right").alias("pairs")
)
```

```text
[
  [
    {"left_value": 1, "right_value": "a"},
    {"left_value": 1, "right_value": "b"},
    {"left_value": 2, "right_value": "a"},
    {"left_value": 2, "right_value": "b"},
  ]
]
```

Indexes and custom field names:

```python
df.select(
    pl.col("a")
    .list.combinations(
        left_index="i",
        right_index="j",
        left_value="x",
        right_value="y",
    )
    .alias("pairs")
)
```

## Nulls

If the input list for a row is null, the result for that row is null.

If a list is empty, the result is an empty list. If it has one item, the result
contains one self-pair.

By default, null values inside lists are kept:

```text
[null, 2].combinations()
→ [
  {"left_value": null, "right_value": null},
  {"left_value": null, "right_value": 2},
  {"left_value": 2, "right_value": 2},
]
```

Set `skip_null=True` to omit pairs where either value is null:

```python
df.select(
    pl.col("a")
    .list.combinations(skip_null=True)
    .alias("pairs")
)
```

`skip_null` filters pair rows only. It does not turn the whole output list into
null. When index fields are emitted, they keep indexes from the original lists.

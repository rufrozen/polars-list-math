# Nested join

`nested_join` attaches values from a flat table to nested list positions without
exploding the left table. Keys may belong to different ancestors of each item.
The function preserves every left row, list boundary, empty/null list, repeated
item and original position.

```python
import polars as pl
from polars_list_math import nested_join

left = pl.DataFrame({"query": ["phone"], "sku": [[[30, 10], [], [20, 30]]]})
right = pl.DataFrame({"query": ["phone", "phone"], "sku": [10, 30], "label": [1, 3]})

result = nested_join(
    left,
    right,
    on={"query": "query", "sku": "sku"},
    shape="sku",
    values={"label": "label"},
)
# label: [[3, 1], [], [None, 3]]
```

## Arguments and validation

| Argument | Meaning |
| --- | --- |
| `left`, `right` | DataFrame or LazyFrame inputs |
| `on` | Required non-empty mapping: left key column → right key column |
| `shape` | Required left List column defining the output's positions and nesting |
| `values` | Required non-empty mapping: right value column → new output column |
| `skip_null_keys=False` | Raise on null key components; opt in to skipping them |

All left columns are retained. Output names must be unique and must not already
exist in left. A missing column, incompatible list structure, unsupported key
type or mismatched key dtype is an error.

Supported scalar key types are Boolean, String, Binary, signed integers
Int8/16/32/64/128 and unsigned integers UInt8/16/32/64. Left List wrappers locate
keys, while their leaf type must exactly match the corresponding right dtype.
Float, temporal, categorical, decimal and container keys are not supported.
The caller performs any conversions before calling the helper.

Right keys must be unique across the entire supplied right table, including
keys not used by left. Identical duplicate payloads still raise an error. There
is no automatic aggregation or first/last selection. Repetition on the left is
allowed and repeats the matched value.

With `skip_null_keys=True`, right records with any null key component are
discarded, and existing left positions with null key components receive null
values. No left positions are removed. A null ancestor key affects all its
existing leaf positions. Null lists have no leaf positions and are preserved
regardless of the flag. Missing matches and matched null payloads produce null
at the corresponding position in either mode.

Payload dtypes are retained, including nested List/Struct values, fixed-size
Arrays, Decimal, categorical/enum and temporal values. If shape has
depth `d` and a payload has dtype `V`, its output dtype is `List` wrapped `d`
times around `V`. All output columns share one search over the keys.

## Ancestor keys

For a four-level dialog structure, keys may have depths 1, 2, 3 and 4:

```python
result = nested_join(
    dialogs,
    labels,
    on={
        "conversation": "conversation_id",
        "message": "message_id",
        "block": "block_id",
        "sku": "target_sku",
    },
    shape="sku",
    values={"grade": "label"},
)
```

At position `[attempt, message, block, item]`, the lookup key uses each
ancestor's value at its corresponding prefix. A scalar key applies to the
whole row. Keys cannot be deeper than shape, and common list boundaries must
align. Shape leaf values are not themselves keys unless shape is included
in `on`. Absent shape branches are preserved without traversing their children.

SKU and canonical annotations can be joined independently using different
shape columns and output names. Normalize block identifiers and their dtypes
in application code before joining.

## Lazy execution

Two DataFrames return a DataFrame. Any LazyFrame input returns a LazyFrame;
constructing the plan only inspects schemas and never collects input data.
Type annotations expose all four combinations.

Execution materializes both inputs. Each input is packed into one List[Struct]
row; a single batch callback passes the entire left frame and one shared right
dictionary to the expression. The Rust expression builds the index once per
invocation. It does not duplicate the dictionary per left row or explode left.

The callback blocks predicate, projection and slice pushdown across the join
so that unused or unmatched right records are still validated. This initial
implementation is not a streaming or out-of-core join. Memory must accommodate
both inputs, the right index and the output. Serialization of plans containing
Python callbacks is subject to Polars' own limitations.

Validation occurs when the join executes. As with other lazy expressions,
Polars can eliminate an entire unused subplan (for example, a final `head(0)`
or constant-false filter); in that case its runtime checks do not execute.
An empty or filtered-empty left input passed **into** the join still validates
the complete right table.

## Public expression: nested_lookup

`nested_lookup(right, *, on, shape, values, skip_null_keys=False)` returns a
Polars Expr. `right` is a column name or expression with dtype
`List[Struct[key fields, value fields]]`. It can contain one dictionary per left
row or a single dictionary shared across the batch.

The result is always a Struct of nested columns, even for one payload:

```python
from polars_list_math import nested_lookup

dictionary = right.to_struct().implode()  # A one-row Series of List[Struct].
result = left.with_columns(
    nested_lookup(
        pl.lit(dictionary),
        on={"query": "query", "sku": "sku"},
        shape="sku",
        values={"label": "label"},
    ).alias("joined")
).unnest("joined")
```

Unlike the table helper, the expression only validates dictionaries actually
passed to it. Output field names must be unique, while the caller controls the
expression's outer alias and handles surrounding column-name collisions.
A null dictionary is an error; use a typed empty list for an empty dictionary.

Both APIs use the Rust plugin when built and a Python fallback otherwise.
Fallback has the same schema, values, ordering and validation rules, including
typed empty and all-null output. Neither path explodes the left table.

See the [runnable example](../examples/nested_join.py)
and [performance comparison](../performance/native_plugins/nested_join/README.md).

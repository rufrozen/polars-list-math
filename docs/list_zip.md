# List zip

`list_zip(*exprs, pad=False, broadcast=False, fields=None)` combines list inputs
into a list of structs for each row. The same operation is available through
`Expr.list.zip` after importing `polars_list_math`.

```python
import polars as pl
from polars_list_math import list_zip

frame = pl.DataFrame({"a": [[1, 2]], "b": [[10]]})
frame.select(list_zip("a", "b", fields=["a", "b"]).alias("zipped"))
# [[{"a": 1, "b": 10}]]

frame.select(list_zip("a", "b", pad=True).alias("zipped"))
# [[{"a": 1, "b": 10}, {"a": 2, "b": None}]]
```

Provide at least two inputs. Strings refer to column names. Struct field names
come from the input expressions unless `fields` supplies a unique name for each
input. By default, inputs must be lists and the shortest list determines the
output length. Set `pad=True` to pad shorter lists with nulls instead.

## Broadcasting scalar values

Set `broadcast=True` to repeat scalar columns or literals within each row:

```python
frame = pl.DataFrame({"items": [[1, 2], [3]], "currency": ["RUB", "USD"]})
frame.select(list_zip("items", "currency", broadcast=True).alias("zipped"))
# [[{"items": 1, "currency": "RUB"}, {"items": 2, "currency": "RUB"}],
#  [{"items": 3, "currency": "USD"}]]

# For a constant string, use pl.lit; bare strings refer to columns.
frame.select(list_zip(
    "items", pl.lit("RUB"), broadcast=True, fields=["item", "currency"]
))
```

The default is `broadcast=False`. With broadcasting enabled, at least one input
must have `List` dtype. Only lists determine the result length: the shortest list
with `pad=False`, or the longest with `pad=True`. Shorter lists are padded with
nulls; scalar values repeat across the entire result. A scalar null becomes a
null struct field, whereas a null list produces a null output row. An empty
result list remains empty. This option is also available on `Expr.list.zip`.

See [the runnable examples](../examples/list_zip.py) for padding and null literals.

Related [Polars proposal](https://github.com/pola-rs/polars/issues/22719).

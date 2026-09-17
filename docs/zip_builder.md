# ZipBuilder

`ZipBuilder` composes `list_zip` and Polars list expressions for nested list
columns. It lives in `polars_list_math/_zip_builder.py` and is exported from
`polars_list_math`.

```python
import polars as pl
from polars_list_math import ZipBuilder

frame = pl.DataFrame({"a": [[[1, 2], [3]]], "b": [[[10, 20], [30]]]})
element = pl.element()

expr = (
    ZipBuilder("a", "b")                 # outer zip
    .level(pad=True)                      # zip the inner lists
    .filter(element.struct.field("a") > 1)
    .eval(element.struct.field("a") + element.struct.field("b"))
    .build("result")
)

frame.with_columns(expr)
# result: [[22], [33]]
```

## Levels and operations

The constructor configures the outermost zip. Each `.level()` descends one
more list level. For `List(List(T))` columns, use one `.level()`; for
`List(List(List(T)))`, use two. There is no fixed depth limit in the builder;
Polars' own expression and dtype limits still apply. Depth is explicit because
`build()` does not have a DataFrame schema or inspect data.

Each zip initially produces a list of structs. At the next depth, the struct
fields become the inputs to another zip. With two zip levels, the result is
`List(List(Struct(...)))`, unless an eval changes the element type.

| Method | Behavior |
| --- | --- |
| `ZipBuilder(*exprs, pad=False, broadcast=False, fields=None)` | Configure the outer zip; at least two inputs |
| `.level(pad=False, broadcast=False, fields=None)` | Add an inner zip over the current struct fields |
| `.eval(expression, parallel=False)` | Transform elements at the current level |
| `.filter(predicate)` | Keep elements where the predicate is true; discard false/null |
| `.build(name=None)` | Return a `pl.Expr`, optionally naming the output column |

Eval and filter may be repeated and run **in call order, before descending to
the next level**. For example, an outer filter removes whole groups, whereas
an inner filter removes zipped records within each group. An eval followed by
a filter makes the filter see the transformed values.

Expressions inside eval/filter use `pl.element()`. Initially an element is a
zipped struct, so access a field with `pl.element().struct.field("name")`.
After projecting a scalar in eval, `pl.element()` refers to that scalar.
Named DataFrame columns are rejected inside these operations. To carry row
metadata into a level, add it as a zip input and enable broadcast.

If an eval is followed by `.level()`, preserve the current struct fields and
their names: the next zip reads those fields. After descending, fluent methods
configure that new depth; configure outer transformations before `.level()`.
Polars checks actual data types when the expression is planned or executed.

## Zip options

Each level independently accepts the existing zip options:

- `pad=False` truncates to the shortest list; `pad=True` pads to the longest.
- `broadcast=True` repeats scalar inputs beside list elements. At least one
  input at that level must be a list. Only lists determine the zip length.
- `fields` names the zipped struct fields in input order. At the outer level,
  names default to the input expression names. Inner levels retain their
  parent's names unless explicitly renamed. Names must be unique.

`pad` and `broadcast` default to false at **every** level; they are not inherited.
Renaming inner fields does not change which parent fields supply the inputs.

```python
expr = (
    ZipBuilder("groups", "currency", broadcast=True, fields=["items", "currency"])
    .level(broadcast=True, fields=["item", "currency"])
    .build("offers")
)
```

For `groups = [[1, 2], [3]]` and scalar `currency = "RUB"`, this yields nested
lists of `{item, currency}` structs. Strings passed as constructor inputs are
column names; use `pl.lit("RUB")` for a string literal.

Null lists propagate through zip; null scalar fields are broadcast normally.
Outer padding creates null fields. If such a field is a list used by the next
zip, that inner result is null. Empty lists retain their nesting. The existing
zip fallback may infer wider scalar dtypes than the native implementation.

## Building and reuse

Fluent methods mutate and return the same builder. `build()` does not consume
it, execute data, or change its configuration. An expression already built is
unaffected by later builder changes. Assign separate builders for independent
configurations.

Without a name, the output uses the outer zip's first field name. The result is
an ordinary expression usable in `select`, `with_columns`, and lazy queries.
Each zip uses the existing native plugin when available, or its Python fallback.
Eval/filter are composed as Polars expressions.

See the runnable [example](../examples/zip_builder.py).

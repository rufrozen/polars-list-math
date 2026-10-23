# Examples

Each file is a standalone example of one part of `polars-list-math`:

- [`nested_join.py`](nested_join.py) attaches flat annotations using row,
  ancestor and leaf keys without exploding nested left columns;
- [`zip_builder.py`](zip_builder.py) zips nested lists with per-level
  transformations, filtering, and scalar broadcasting;
- [`list_zip.py`](list_zip.py) zips list columns into a list of structs, with optional scalar broadcasting;
- [`combinations.py`](combinations.py) builds pairs within one list or between
  two lists;
- [`similarity.py`](similarity.py) compares ordered lists in Polars and plain
  Python;
- [`kendall_tau.py`](kendall_tau.py) compares the order of common IDs, with
  optional normalization and means against peers or references;
- [`containment.py`](containment.py) compares SERPs using directed set containment
  and averages within groups or against references;
- [`mean_similarity.py`](mean_similarity.py) compares lists inside nested-list
  columns.
- [`expected_value_of_game.py`](expected_value_of_game.py) computes the expected
  value of a sequential game;
- [`json_object_items.py`](json_object_items.py) converts JSON objects into
  lists of key/value structs;
- [`json_array_values.py`](json_array_values.py) converts JSON arrays into
  lists of strings;
- [`url_query_encode.py`](url_query_encode.py) encodes Struct and key/value-list
  expressions as query strings;
- [`url_build.py`](url_build.py) assembles URLs from optional component
  expressions;
- [`null_schema.py`](null_schema.py) discovers absent nested fields, cleans
  their schema, and applies it to a LazyFrame.

Install the project, then run an example from the repository root:

```bash
make develop
uv run python examples/list_zip.py
uv run python examples/zip_builder.py
uv run python examples/containment.py
uv run python examples/json_object_items.py
uv run python examples/url_build.py
```

The examples use the typed top-level expression functions so Pyright and
Pylance can validate them. Importing `polars_list_math` also registers equivalent
runtime shorthand methods on `Expr.list` and `Expr.str`.

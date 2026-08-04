# Examples

Each file is a standalone example of one part of `polars-list-math`:

- [`list_zip.py`](list_zip.py) zips list columns into a list of structs;
- [`combinations.py`](combinations.py) builds pairs within one list or between
  two lists;
- [`similarity.py`](similarity.py) compares ordered lists in Polars and plain
  Python;
- [`mean_similarity.py`](mean_similarity.py) compares lists inside nested-list
  columns.
- [`json_object_items.py`](json_object_items.py) converts JSON objects into
  lists of key/value structs;
- [`json_array_values.py`](json_array_values.py) converts JSON arrays into
  lists of strings;
- [`url_query_encode.py`](url_query_encode.py) encodes Struct and key/value-list
  expressions as query strings;
- [`url_build.py`](url_build.py) assembles URLs from optional component
  expressions.

Install the project, then run an example from the repository root:

```bash
make develop
uv run python examples/list_zip.py
uv run python examples/json_object_items.py
uv run python examples/url_build.py
```

Imports of `polars_list_math` are intentional: importing the package registers
the additional methods on `Expr.list` and `Expr.str`.

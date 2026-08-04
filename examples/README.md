# Examples

Each file is a standalone example of one part of `polars-list-math`:

- [`list_zip.py`](list_zip.py) zips list columns into a list of structs;
- [`combinations.py`](combinations.py) builds pairs within one list or between
  two lists;
- [`similarity.py`](similarity.py) compares ordered lists in Polars and plain
  Python;
- [`mean_similarity.py`](mean_similarity.py) compares lists inside nested-list
  columns.

Install the project, then run an example from the repository root:

```bash
make develop
uv run python examples/list_zip.py
```

The import below is intentional in every example: importing
`polars_list_math` registers the additional methods on `Expr.list`.

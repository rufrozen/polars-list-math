# Polars DataFrame construction

This isolated test suite compares construction of a Polars `DataFrame` from:

- a regular `dataclass`;
- a `dataclass(slots=True)`;
- a `NamedTuple`.

Every row has exactly 30 columns and contains scalar columns, a nested struct,
a list of structs, and regular lists. Both a single-row frame and an `N`-row
frame are tested. The default `N=10_000` keeps the complete suite comfortably
below one minute on a typical development machine.

Run from the repository root:

```bash
uv run pytest performance/dataframe_construction -v -s
```

Override the number of rows when needed:

```bash
POLARS_CONSTRUCTION_ROWS=25000 uv run pytest performance/dataframe_construction -v -s
```

The tests print the construction time for every representation and size. The
one-minute requirement applies to the default configuration; increasing `N`
may naturally make the run longer.

## Results

Control run on 2026-08-19:

- Python 3.12.3;
- Polars 1.42.1;
- pytest 9.1.1;
- 10,000 rows and 30 columns;
- values are generated before the timed section, so the table measures only
  `pl.DataFrame(...)` construction.

| Input representation | 1 row | 10,000 rows |
|---|---:|---:|
| `dataclass` | 0.0227 s | 0.9897 s |
| `dataclass(slots=True)` | 0.0003 s | 0.8201 s |
| `NamedTuple` | 0.0002 s | 0.4148 s |

The complete test run took **2.78 seconds** (`7 passed`), well below the
one-minute limit. The first one-row measurement includes Polars initialization,
so the 1-row values should not be treated as a stable microbenchmark. Repeat
the test or use a dedicated warm-up when comparing very small inputs.

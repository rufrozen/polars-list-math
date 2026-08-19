# typed_polars construction performance

This experiment measures complete DataFrame construction from two equivalent,
already generated datasets:

- public `typed_polars` slots models through `TypedRow.to_frame_many()`;
- ready outer and nested `NamedTuple` objects passed directly to
  `pl.DataFrame()`.

Both datasets contain the same 19 physical columns and cover:

- scalar fields;
- a nested Struct;
- a Struct inside another Struct;
- a List[Struct];
- a flat Struct expanded into scalar columns;
- a flat List[Struct] expanded into parallel list columns;
- regular lists;
- a typed dictionary stored as List[Struct[key, value]];
- a nullable field.

Dataset generation is deliberately outside the timed section. The
`typed_polars` measurement therefore includes compilation of its physical
plan, conversion from slots dataclasses to tuples/nested named tuples, and
Polars construction. The ready-NamedTuple baseline measures only Polars
construction and represents the lowest-overhead input expected from this
design.

Run from the repository root:

```bash
uv run pytest performance/typed_polars_construction -v -s
```

Override the default 10,000 rows:

```bash
POLARS_TYPED_ROWS=25000 uv run pytest performance/typed_polars_construction -v -s
```

## Results

Control run on 2026-08-19 with Python 3.12.3 and Polars 1.42.1:

| Input | 1 row | 10,000 rows |
|---|---:|---:|
| `typed_polars` slots models | 0.0011 s | 0.4315 s |
| ready nested `NamedTuple` | 0.0010 s | 0.4447 s |

The complete suite took **1.10 seconds** (`4 passed`). At 10,000 rows the full
`typed_polars` path was about 3% faster than the ready-NamedTuple baseline in
this run; the difference is small enough to treat the implementations as
roughly equivalent without repeated benchmark runs. The one-row result is not
a stable microbenchmark.

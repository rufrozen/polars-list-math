# Nested Struct input representations

This experiment checks which Python values Polars 1.42.1 accepts as nested
`Struct` and `List[Struct]` values when the outer DataFrame row is a tuple.

The tested representations are:

- dictionaries;
- ordinary positional tuples;
- `NamedTuple` instances;
- `dataclass(slots=True)` instances.

Run from the repository root:

```bash
uv run pytest performance/nested_struct_input -v -s
```

## Results

Control run on 2026-08-19 with Python 3.12.3 and Polars 1.42.1:

| Nested representation | Inferred schema | Explicit schema | Data preserved |
|---|---:|---:|---:|
| `dict` | yes | yes | yes |
| ordinary `tuple` | no | no | n/a |
| `NamedTuple` | yes | yes | yes |
| `dataclass(slots=True)` | no | construction succeeds | **no** |

An ordinary tuple has no field names, and Polars does not map its positions to
Struct fields even when an explicit schema is supplied. A `NamedTuple` does
carry field names and works for both a single Struct and items inside a
List[Struct].

The slots-dataclass result with an explicit schema is particularly unsafe in
this Polars version: construction succeeds but nested values become null. The
test records this current behavior as a known limitation rather than treating
it as a supported conversion.

The suite also prints construction timings for 10,000 rows using dictionaries
and named tuples. Dataset generation is outside the timed section. Timings are
informational and can vary by machine.

| Representation | 10,000 rows |
|---|---:|
| `dict` | 0.0476 s |
| `NamedTuple` | 0.0321 s |

The complete control run took **0.18 seconds** (`8 passed`). In this run the
nested `NamedTuple` input was about 1.5 times faster than dictionaries during
DataFrame construction.

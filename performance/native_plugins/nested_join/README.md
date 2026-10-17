# Nested join: native, Python and explode/implode

This experiment compares full eager operations on identical inputs:

- Rust `nested_join`, with one shared right dictionary and index per call.
- Python fallback, with the same schema, validation and nested traversal.
- Polars `explode → left join → implode`, restoring all levels by existing
  positional indices and explicitly preserving left order.

Each strategy must produce exactly the same values, dtypes, row order and
nested structure before timing begins. Empty lists, null lists and null leaf
positions remain distinct. There are no machine-dependent speed thresholds.

## Workloads

| Scenario | Shape depth | Keys | Characteristics |
| --- | ---: | --- | --- |
| `serp` | 2 | query, SKU | Several short search results per row |
| `dialog` | 4 | conversation, message, block, SKU | Ancestor keys at four different depths |
| `skew` | 4 | conversation, message, block, SKU | Some item lists are ten times longer |

Data is deterministic and ragged, with repeated left keys, repeated contexts,
empty/null branches and missing matches. Every workload transfers two fields:
an Int8 label and a Float64 score. Right keys are unique; invalid-input behavior
is covered by the regular test suite. `--rows` controls input size.

Row and per-level positional indices are supplied to **all three strategies**
before measurement. Their creation, input generation and client dtype
normalization are excluded. Every measured call includes validation, right
index construction, lookup and output construction. The reference additionally
includes explode, join, grouping and reconstruction. No expanded left table,
hash index or completed result is reused between calls.

## Run

Build the native extension first, then run from the repository root:

```bash
POLARS_MAX_THREADS=4 uv run --no-sync --group dev \
  pytest performance/native_plugins/nested_join -v -s

POLARS_MAX_THREADS=4 uv run --no-sync --group dev \
  python -m performance.native_plugins.nested_join.benchmark \
  --scenario skew --rows 20000 --repeats 3
```

The pytest entry point runs all three scenarios. The standard
`POLARS_LIST_MATH_PERF_ROWS` and `POLARS_LIST_MATH_PERF_REPEATS` environment
variables also apply. These experiments are excluded from `make test`.

Each strategy runs in a separate subprocess, with one warm-up and three measured
repeats by default. Input data stays resident; the previous output is released
before the next call. The primary time is the minimum; JSON also records the
median and leaf positions processed per second.

On Linux, worker `/proc/self/status` supplies baseline VmRSS after input
preparation and peak VmHWM after all calls. Reported growth is their difference.
Peak includes imports, input preparation, warm-up, outputs and allocator
retention across repeats; it is **not** an exact count of temporary allocations.
Unlike inherited process resource counters, VmHWM resets on exec. Memory metrics
are null on platforms without `/proc/self/status`.

## Recorded results

Measured 2026-10-17 on AMD Ryzen 7 PRO 8840U, WSL2 / Linux x86_64,
Python 3.12.13, Polars 1.42.1, Rust 1.96.1 and maturin 1.14.1.
`POLARS_MAX_THREADS=4`; one warm-up and three measured repeats.
The extension used an optimized dev build (`CARGO_PROFILE_DEV_OPT_LEVEL=3`,
`CARGO_PROFILE_DEV_DEBUG=0`), not the default unoptimized dev profile.

| Scenario | Left rows | Leaf positions | Right rows | Native, ms | Fallback, ms | Explode, ms | Fallback/native | Explode/native |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| serp | 2,000 | 47,674 | 3,000 | 7.31 | 74.25 | 19.06 | 10.15x | 2.61x |
| dialog | 2,000 | 47,035 | 8,962 | 10.37 | 172.18 | 38.95 | 16.61x | 3.76x |
| skew | 2,000 | 107,344 | 38,916 | 25.33 | 308.40 | 45.14 | 12.18x | 1.78x |
| skew | 20,000 | 1,076,341 | 390,509 | 361.81 | 3921.79 | 305.73 | 10.84x | 0.84x |

A ratio above 1 favors native. Native is faster than explode on the three
smaller cases, but is about 18% slower on the largest skewed case.
The Rust lookup uses a single index/traversal; this result does not imply
that avoiding explode always beats Polars' parallel join/grouping kernels.

| Scenario / rows | Native peak / growth, MiB | Fallback peak / growth, MiB | Explode peak / growth, MiB |
| --- | ---: | ---: | ---: |
| serp / 2,000 | 78.9 / 13.3 | 74.9 / 9.4 | 90.4 / 24.8 |
| dialog / 2,000 | 81.7 / 11.8 | 88.5 / 18.7 | 108.1 / 37.1 |
| skew / 2,000 | 107.1 / 21.0 | 120.9 / 35.4 | 137.1 / 51.8 |
| skew / 20,000 | 507.6 / 257.9 | 614.1 / 369.1 | 664.8 / 414.4 |

On the largest case, native processes 2.97 million leaf positions/second
and has about 24% lower measured peak RSS than explode. Peak and growth
use the process-level method described above; input-generation retention
and native-library loading can dominate small cases.

All twelve measurements, medians, throughputs, baselines and environment
metadata are available in [results.json](results.json).

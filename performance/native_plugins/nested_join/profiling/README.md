# Native nested join bottlenecks

Measured against `4662fd3` on 2026-10-17. The main bottleneck is the left
traversal and composite-key lookup, followed by construction of the right
index. Three experimental changes reduced the large workload's median from
442.53 to 328.80 ms in a final comparison without profiling timers. The combined
optimizations are now applied to the library. The measurements below preserve
the original investigation; runtime diagnostics were removed
from the applied implementation.

## Workload and method

Same host and optimized dev build as the [parent experiment](../README.md):
Ryzen 7 PRO 8840U, WSL2/Linux, Python 3.12.13, Polars 1.42.1, Rust 1.96.1,
maturin 1.14.1, `CARGO_PROFILE_DEV_OPT_LEVEL=3`, `CARGO_PROFILE_DEV_DEBUG=0`.
The large `skew` case has 20,000 left rows, 1,076,341 leaf positions, 390,509
right records, four keys at different depths and two numeric payloads.

Every timed call rebuilds its index and includes validation, lookup and output
construction. Positional indices already exist, as in the original experiment.
One warm-up per variant checks exact equality against explode, including
dtypes, order, missing matches, nulls and empty lists. Five measured rounds
alternate the order of variants. Inputs are shared in one process; explicit
garbage collection and equality checks are outside timing. Each binary has a
separate immutable path to avoid the plugin loader's library cache.

Medians are used here to compare typical calls. Raw repetitions and minima
are in [results.json](results.json). WSL scheduling/CPU frequency introduces
visible variance; these observations are not confidence intervals or speed
guarantees. Do not compare these medians directly with the parent experiment's
historical minima from separate processes. Timing runs did not run concurrently.

The coarse Rust timers surround whole stages, with no timer per leaf. In a
paired overhead check, the unchanged library measured 424.21 ms and its
instrumented version 424.58 ms. Timers do not explain the prototype speedup.
No hardware performance counters were available: cache effects discussed
below are hypotheses, not measured cache-miss counts.

## Where time goes

Instrumented baseline, medians from the overhead-check run:

| Stage | Time, ms | Approximate share of Rust time |
| --- | ---: | ---: |
| Walk left shape and probe composite keys | 254.18 | 60% |
| Build and validate right index | 129.95 | 31% |
| Gather payloads and wrap list levels | 24.36 | 6% |
| Drop right index | 8.23 | 2% |
| Schema, rechunking and setup | 1.11 | <1% |
| Construct final Struct | 0.04 | <1% |

Rust total was 421.55 ms versus 424.58 ms for the complete call. The residual
also includes FFI and cleanup; it is not exclusively Python execution time.
Stage medians are calculated independently and need not sum to the median
total. Optimizing the Python wrapper or final Struct is not the first priority
for this workload. Fragmented inputs and lazy execution require separate
measurements; these numbers do not establish their costs.

The code at baseline commit `4662fd3` explains the expensive work:

- [`Walker::walk`](../../../../rust/src/nested_join.rs) reads every ancestor
  key again at each leaf, dispatches on its physical dtype and downcasts the
  array again, then hashes the entire composite key. It also allocates a
  `key_starts` vector at every non-null list node and a positions vector per
  top-level row. Its probe vector is **already reused** in the baseline.
- `build_index` allocates one `Vec<KeyPart>` per right record. On this build
  each part occupies 32 bytes. Four parts times 390,509 records account for
  about 47.7 MiB of key buffers alone, excluding the hash table, allocation
  metadata and borrowed string bytes. This is a layout calculation, not RSS.
  The baseline **already uses** `PlHashMap` and an entry-based insertion.
- Payload construction calls a dynamic Arrow builder once per leaf per
  transferred column. The workload makes over two million tiny operations.
  List offsets and validity buffers are already reused during wrapping.
- Index construction and left traversal are sequential. Increasing Polars'
  thread count does not parallelize those loops.

## Isolated prototypes

These measurements come from one alternating comparison with four Polars
threads. Individual changes are relative to the same instrumented baseline;
the original, uninstrumented binary was also measured in this run.

| Variant | Full call median, ms | Build, ms | Walk, ms | Gather, ms |
| --- | ---: | ---: | ---: | ---: |
| Original without timers | 429.66 | — | — | — |
| Original with timers | 421.12 | 127.35 | 257.33 | 25.27 |
| Bulk gather | 403.71 | 131.63 | 246.15 | 9.60 |
| Reuse buffers and ancestor values | 377.44 | 140.43 | 195.86 | 28.11 |
| Compact index | 403.59 | 85.82 | 291.11 | 26.55 |
| All three changes | 335.18 | 84.41 | 242.80 | 9.87 |
| Explode/join/implode | 425.66 | — | — | — |

Buffer reuse is the best isolated improvement here.
It retains per-depth scratch buffers and rereads an ancestor value only when
its physical position changes. It still hashes the whole tuple, still uses
dynamic key readers when positions change, and still allocates positions per
top-level row. It does not implement prefix hashing.

Bulk gather creates one `IdxSize` index buffer and calls
`opt_gather_extend` once per payload. An out-of-bounds sentinel produces a
null. The old scalar path remains for dictionaries that exceed `IdxSize`
capacity. This speeds up the gather stage by about 2.6x, but that stage is
only about 6% of the baseline. The prototype adds a compact gather buffer;
it does **not** replace the existing `Vec<Option<usize>>` match buffer.

Compact indexing stores a hash and right row number,
using the original columns for exact equality and an explicit collision
chain. A hash match alone is never accepted. This removes the per-record
key-buffer allocation and almost eliminates index destruction time, but
rereading right columns makes probing slower: 257 → 291 ms. Its end-to-end
benefit is smaller than its index-build improvement suggests. This design
could benefit from further comparison against pooled or row-encoded keys.

The combined prototype included all three changes. The compact-index
prototypes used a diagnostic `PLM_LOOKUP_HASH_BITS` hook and a capacity
accessor for profiling. Both were removed from the applied implementation.
The prototype hook forced every hash to collide when set to zero; the library
has a deterministic Rust unit test of colliding
composite keys, missing matches and duplicate rejection within a collision
chain. No runtime environment switch is needed.

## Confirmation without timers

Four threads, a freshly rebuilt combined prototype with all stage timers
removed, measured together with the unchanged binary and explode:

| Scenario / left rows | Original median, ms | Combined median, ms | Explode median, ms |
| --- | ---: | ---: | ---: |
| serp / 2,000 | 7.13 | 6.99 | 22.27 |
| dialog / 2,000 | 12.65 | 11.66 | 44.31 |
| skew / 2,000 | 29.76 | 24.38 | 50.93 |
| skew / 20,000 | 442.53 | 328.80 | 380.53 |

For the large case this is 25.7% less time than native baseline and 13.6% less
than explode in the same run. Minima were 408.14 / 309.06 / 342.67 ms,
respectively. The small SERP difference is too small to claim a clear gain.

In separate processes, the large case's peak VmHWM was 504.5 MiB for baseline
and 504.1 MiB for combined; initial VmRSS was 246.0 and 245.4 MiB. No meaningful
peak-RSS improvement was demonstrated. These are process high-water marks,
including data generation, imports, library loading and allocator retention,
not live index sizes. Memory workers used one warm-up and five calls and
sampled before constructing the equality reference. Their timings are not
used to claim an additional speedup.

A separate one-thread comparison measured baseline / combined / explode at
404.88 / 342.10 / 640.96 ms. The corresponding earlier four-thread comparison
was 429.66 / 335.18 / 425.66 ms. Combined in these thread comparisons includes
timers. The native loops show no useful scaling; explode benefits from Polars
parallelism. These were separate processes/runs, not controlled scaling
estimates, so their ratios should not be treated as precise speedups.

## Applied changes and remaining work

The applied implementation was rebuilt after removing the runtime collision
hook and unused profiling accessor, and adding a testable insertion method.
The final comparison used the same optimized build, four threads, one warm-up
and five alternating measured rounds, with no concurrent compilation:

| Scenario / left rows | Baseline median, ms | Applied median, ms | Explode median, ms |
| --- | ---: | ---: | ---: |
| serp / 2,000 | 7.50 | 7.05 | 22.56 |
| dialog / 2,000 | 12.02 | 11.01 | 46.05 |
| skew / 2,000 | 26.06 | 22.02 | 57.08 |
| skew / 20,000 | 452.17 | 316.12 | 383.16 |

The large case takes 30.1% less time than baseline and 17.5% less than explode
in this final run. Exact output equality passed for all workloads. Full
validation passed: 374 Python tests (native and fallback), one Rust collision
test, Ruff and Pyright. Raw repetitions, environment and the source digest are
in [applied_results.json](applied_results.json). The same variability and
workload limitations described above apply.

Implementation status and follow-ups:

1. Traversal scratch space and ancestor values are reused, preserving shape
   checks and null-key errors. Reusing the remaining per-row positions vector
   is a follow-up.
2. Bulk payload gathering is applied. Investigate writing compact match indices
   directly: the current 16-byte `Option<usize>` costs about 16.4 MiB for this
   case, versus about 4.1 MiB for 32-bit indices with a reserved null sentinel.
   This potential saving is **not** implemented in the library. Retain a
   wide-index path when needed.
3. The compact index is applied with exact equality and collision chains.
   Compare alternatives: preselect typed key readers once per column and test
   pooled/row-encoded keys to avoid expensive right-column rereads. Measure
   construction and probing separately.
4. Cache hashes of ancestor prefixes for long repeated contexts. The prototype
   caches values only; it still rehashes every string at every leaf. Left and
   right must use the same key ordering and encoding, including arbitrary
   order in `on`. This is an unmeasured follow-up, not a promised gain.
5. Parallelize disjoint top-level row ranges over one immutable validated
   shared index. Partition by estimated leaf work, not just row count, to
   handle skew. Keep worker scratch separate and writes disjoint. A parallel
   prototype and crossover threshold are still needed.

Do not mark the expression elementwise or duplicate the shared right
dictionary per left row to obtain parallelism. Do not skip full-right
validation, accept hash equality, cache a previous call's index in a benchmark,
or drop null positions. All would alter semantics or the measured operation.

The 112 nested-join tests passed for each instrumented prototype. Compact and
combined also passed 56 native cases with every hash forced to collide. The
uninstrumented combined prototype passed the original 364-test suite. The
applied implementation adds regression cases for ancestor cache changes across
skipped keys and rows, both key orders, complex/sliced/empty payload sources,
and distinct row-local dictionaries. Run `make test` for native and fallback
cases and `make test-rust` for the deterministic collision-chain test. Additional
payload, fragmented-input and lazy benchmarks are still needed before claiming
general performance improvements for every supported dtype or execution mode.

## Reproduce

Compare the baseline implementation at `4662fd3` with the current optimized
implementation. Work in a disposable checkout of the current commit with its
own development environment to keep experimental source and binaries out of
the regular installation.

Build baseline with the same optimization settings and copy its shared
library into an empty snapshot directory. Keep snapshot files immutable:

```bash
export CARGO_PROFILE_DEV_OPT_LEVEL=3
export CARGO_PROFILE_DEV_DEBUG=0
export CARGO_BUILD_JOBS=1
export POLARS_MAX_THREADS=4
export PLM_PROFILE_LIBRARIES="$(mktemp -d)"

# In the disposable checkout, restore the historical implementation first.
git show 4662fd3:rust/src/nested_join.rs > rust/src/nested_join.rs
make develop
mkdir "$PLM_PROFILE_LIBRARIES/baseline"
cp polars_list_math/_native*.so "$PLM_PROFILE_LIBRARIES/baseline/"

# Restore the current implementation before building the optimized library.
git show HEAD:rust/src/nested_join.rs > rust/src/nested_join.rs
make develop
mkdir "$PLM_PROFILE_LIBRARIES/applied"
cp polars_list_math/_native*.so "$PLM_PROFILE_LIBRARIES/applied/"

uv run --no-sync --group dev python \
  -m performance.native_plugins.nested_join.profiling.compare \
  --library-root "$PLM_PROFILE_LIBRARIES" \
  baseline applied explode --rows 20000 --scenario skew --repeats 5
```

The driver emits `PLM_RUN` markers to stderr identifying each variant and
iteration (zero is warm-up), and full-call measurements to stdout. It does not
mutate source files or libraries. Stage timings and isolated-prototype results
are retained as historical measurements in `results.json`; the instrumentation
and separate prototype sources are no longer included.

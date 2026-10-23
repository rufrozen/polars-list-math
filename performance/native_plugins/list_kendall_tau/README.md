# Kendall tau performance

Compare all three Rust Kendall tau expressions against their Python fallback.
The shared runner checks result equality, warms up each plan, and reports the
best of three timed collections. Frame construction and expression creation
are excluded. There are no timing assertions.

The baseline uses eight IDs per list, three target lists and two reference
lists. The complex scenario uses 128 IDs, eight target lists and two reference
lists, with `normalize=True`. Both include null items, null inner lists and
empty inner lists. The right ranking combines the even-positioned IDs with
the reversed odd-positioned IDs.

Run from the repository root with an optimized native extension:

```bash
uv run --no-sync --group dev maturin develop --release
uv run --no-sync --group dev pytest performance/native_plugins/list_kendall_tau -q -s
```

Override `POLARS_LIST_MATH_PERF_ROWS` and `POLARS_LIST_MATH_PERF_REPEATS` to change
the default 2,000 rows and three repeats.

## Recorded results

2026-10-24, Ubuntu under WSL, AMD Ryzen 7 7800X3D, Python 3.12.3,
Polars 1.42.1, Rust 1.96.0, optimized release build. All six native/fallback
result comparisons passed.

| Scenario | Native | Python fallback | Speedup |
| --- | ---: | ---: | ---: |
| `kendall_tau` | 0.002607 s | 0.018817 s | 7.22x |
| `mean_kendall_tau` | 0.004788 s | 0.045482 s | 9.50x |
| `mean_kendall_tau_to` | 0.008512 s | 0.074697 s | 8.78x |
| `kendall_tau_complex` | 0.038279 s | 0.313601 s | 8.19x |
| `mean_kendall_tau_complex` | 0.329232 s | 4.397959 s | 13.36x |
| `mean_kendall_tau_to_complex` | 0.275744 s | 2.974351 s | 10.79x |

Timings depend on hardware, build profile and input distribution. See
[the feature guide](../../../docs/kendall_tau.md) for ranking and null semantics.

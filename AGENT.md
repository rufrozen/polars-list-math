# Agent Guide

This repository is `polars-list-math`, a Python 3.12+/Rust package that adds
list, JSON, and URL helpers to Polars. The distributable Python
package contains pure-Python fallbacks and a maturin-built Polars expression
plugin named `polars_list_math._native`.

## Public API

Importing `polars_list_math` calls `install()` and conditionally adds methods to
the existing Polars namespaces. Existing methods are not overwritten unless
`install(overwrite=True)` is requested.

| Area | Top-level API | Runtime namespace API |
|---|---|---|
| Nested zip builder | `ZipBuilder` | none |
| List zip | `list_zip` (`zip_list` internally) | `Expr.list.zip` |
| Nested join | `nested_join`, `nested_lookup` | none |
| Combinations | `list_combinations`, `list_combinations_to` | `Expr.list.combinations`, `Expr.list.combinations_to` |
| Similarity | `list_similarity`, `list_mean_similarity`, `list_mean_similarity_to` | `Expr.list.similarity`, `Expr.list.mean_similarity`, `Expr.list.mean_similarity_to` |
| Containment | `list_containment`, `list_mean_containment`, `list_mean_containment_to`, `py_list_containment` | `Expr.list.containment`, `Expr.list.mean_containment`, `Expr.list.mean_containment_to` |
| Python similarity | `py_list_similarity` | none |
| Kendall tau | `list_kendall_tau`, `list_mean_kendall_tau`, `list_mean_kendall_tau_to`, `py_list_kendall_tau` | `Expr.list.kendall_tau`, `Expr.list.mean_kendall_tau`, `Expr.list.mean_kendall_tau_to` |
| Sequential game | `expected_value_of_game` | none |
| JSON | `json_object_items`, `json_array_values` | `Expr.str.json_object_items`, `Expr.str.json_array_values` |
| URL | `url_query_encode`, `url_build` | none |
| DataFrame source | `dfs_to_lazy_df` | none |
| Null schema | `find_null_schema`, `clean_schema`, `apply_schema` | none |

Treat the top-level expression functions as the static-analysis API. Pyright
and Pylance cannot augment namespace classes owned by Polars, so dynamically
installed `Expr.list` and `Expr.str` methods are runtime conveniences only.

## Project Layout

### Python expression helpers

- `polars_list_math/__init__.py` exports the public helpers and installs the
  runtime Polars namespace methods.
- `_list_zip.py` implements registration, validation, and fallback behavior for
  `list_zip` / `Expr.list.zip`.
- `_zip_builder.py` composes nested zips with per-level eval/filter operations.
  The constructor configures the outer zip; `level()` descends, and `build()`
  returns an expression without evaluating data.
- `_nested_join.py` provides strict ancestor/leaf key lookup with a Rust plugin
  and Python fallback. Lazy joins pack both inputs into one-row List[Struct]
  containers and execute via a non-streaming batch callback; never duplicate a
  shared right dictionary per left row or drop full-right validation.
- `_list_combinations.py` implements both same-list and cross-list pair
  combinations.
- `_list_similarity.py` implements weighted list similarity, the scalar Python
  API, shared Python scoring helpers, and native-library detection.
- `_list_mean_similarity.py` implements self and reference mean-similarity for
  nested lists.
- `_list_kendall_tau.py` compares the ranks of common IDs and provides self and
  reference means, with optional normalization. Undefined pairs return null
  and are excluded from means. Its batch fallback preserves temporal precision.
- `_list_containment.py` implements directed set containment and its nested-list
  means; empty target sets score zero and null peers are excluded.
- `_expected_value_of_game.py` evaluates forward-only sequential games.
- `_json_object_items.py` and `_json_array_values.py` convert JSON strings to
  nested Polars values and support strict/non-strict parsing.
- `_url_query_encode.py` encodes Struct or key/value-list expressions.
- `_url_build.py` assembles URLs from six optional component expressions.
- `_dfs_to_lazy_df.py` exposes DataFrame batches as a pushdown-aware LazyFrame
  source. It is Python-only.
- `_find_null_schema.py` discovers absent nested leaves with a lazy aggregate;
  `_clean_schema.py` subtracts subschemas and applies the resulting schema
  lazily. Both are Python-only and handle List, Array, and Struct nesting.
- `py.typed` marks the installed package as typed.

Rust-backed Python modules detect a compiled library beside the package. When
it is present they call `polars.plugins.register_plugin_function`; otherwise
they construct a Python `map_elements` or `map_batches` fallback. Keep validation
and observable results aligned between both paths, including nested output
dtypes and UInt32 combination indices. Resolve fallback output types from the
input schema rather than inferring them from sample Python values.

### Rust plugin

- `rust/` is the only Rust crate and is built by maturin through
  `rust/Cargo.toml`. Do not add a root-level Rust crate.
- `rust/src/lib.rs` is only the module aggregator, Polars allocator setup, and
  importable `_native` PyO3 module. Keep it small.
- Feature implementations live in matching files:
  `list_zip.rs`, `list_combinations.rs`, `list_similarity.rs`,
  `list_mean_similarity.rs`, `list_containment.rs`, `list_kendall_tau.rs`, `expected_value_of_game.rs`,
  `json_object_items.rs`, `json_array_values.rs`, `url_query_encode.rs`, and
  `url_build.rs`.
- `nested_join.rs` implements the nested lookup index, path traversal and typed
  payload gathering. `polars-core` enables additional value dtypes and Int128
  keys without enabling Polars I/O features.
  Its compact index checks exact key equality along hash collision chains;
  cached ancestor values and Arrow bulk gathering preserve all left positions.
- `list_similarity_core.rs` contains shared primitive ID normalization, list
  readers, first-occurrence ranks, and similarity scoring. Containment reuses its readers with unit
  weights and computes directed intersection ratios.

When adding a Rust-backed helper, give it a focused module under `rust/src/`,
wire it from `rust/src/lib.rs`, add the Python API/fallback, export it where
appropriate, and add native and fallback tests.

### Tests, docs, examples, and performance

- `tests/` contains the regular pytest suite. Feature tests live in matching
  `test_<feature>.py` files.
- `docs/` contains feature guides and publishing instructions.
- `CHANGELOG.md` records release notes, release status, and upgrade guidance;
  update it when preparing a release.
- `examples/` contains standalone runnable examples indexed by
  `examples/README.md`.
- `performance/` contains isolated pytest performance experiments. Every
  experiment has its own folder and README with its scenario, invocation, and
  recorded control results.
- `performance/native_plugins/` compares every Rust expression plugin with its
  Python fallback. `_benchmark.py` owns common warm-up, timing, result equality,
  and environment handling. Operations have separate folders, with
  combinations and similarity variants grouped together. Each operation has a
  baseline and a complex workload.

Performance tests are deliberately outside the default `testpaths`; `make test`
does not run them. Never add machine-dependent timing thresholds. Check result
equivalence and report timings/speedups instead. When workloads or measurement
logic change, rerun the affected experiment and update its recorded README
results.

## Commands

Use the Makefile targets as the project contract:

```bash
make init       # create/sync the development environment without this project
make install    # sync dependencies without building the current project
make lock       # update uv.lock
make develop    # build/install the editable native extension
make format     # format Python
make lint       # Ruff format/check plus Pyright
make test       # regular tests under tests/
make test-rust  # Rust unit tests, including deterministic lookup hash collisions
make build      # build wheel and source distribution
make check-dist # build and validate distributions with Twine
make package    # create the requested source archive
```

Before finishing a normal code change, run:

```bash
make lint
make test
```

After Rust changes, rebuild the extension before testing:

```bash
cargo fmt --manifest-path rust/Cargo.toml
make develop
make lint
make test
make test-rust
```

For packaging, dependency metadata, or Rust crate-layout changes, also run the
relevant distribution checks:

```bash
make build
make package
# or make check-dist when distribution validation is required
```

Run all native performance experiments with:

```bash
uv run pytest performance/native_plugins -v -s
```

The default is 2,000 rows and three measured repeats. Override it with:

```bash
POLARS_LIST_MATH_PERF_ROWS=10000 \
POLARS_LIST_MATH_PERF_REPEATS=5 \
uv run pytest performance/native_plugins -v -s
```

Other construction experiments are listed in `performance/README.md` and run
by passing their folder explicitly to pytest.

## Tooling and Configuration

- Use `uv` through the Makefile variable `UV_CMD`; normal development commands
  use `uv run --no-sync --group dev`.
- Python dependencies and tool configuration live in `pyproject.toml`;
  `uv.lock` is committed.
- Rust dependencies live in `rust/Cargo.toml`; `rust/Cargo.lock` is committed.
- `ruff` owns Python formatting, import sorting, and linting.
- `pyright` runs in basic mode over `polars_list_math` and `tests`.
- The maturin module name is `polars_list_math._native` and its manifest is
  `rust/Cargo.toml`.

## Change Guidelines

- Prefer small feature-focused changes and preserve the Python/Rust split.
- Keep Python registration/fallback code, Rust implementation, regular tests,
  performance tests, documentation, and examples in feature-specific files.
- Test native and fallback behavior whenever either implementation changes.
- Build expressions while the desired native/fallback path is selected; plugin
  selection happens during expression construction, not collection.
- Keep `list.zip` as one feature within the broader package, not the project
  identity.
- Use pytest-style tests and deterministic input data.
- Do not introduce performance assertions based on wall-clock duration.
- Update public exports, docs, examples, and typing coverage when adding or
  renaming an API.

## Git and Generated Artifacts

Commit source files and both lock files when they change. Do not commit:

- `.venv/`, `.pytest_cache/`, `.ruff_cache/`, or any `__pycache__/`;
- `build/`, `dist/`, root `target/`, or `rust/target/`;
- compiled `polars_list_math/*.so`, `*.pyd`, `*.dll`, or `*.dylib` files;
- locally generated package archives unless explicitly requested.

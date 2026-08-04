# Agent Guide

This repository is `polars-list-math`, a Python/Rust package for
list-oriented math and expression helpers for Polars.

## Project Layout

- `polars_list_math/` contains the Python package.
- `polars_list_math/_list_combinations.py` contains the Python registration and
  fallback path for the `list.combinations` and `list.combinations_to` helpers.
- `polars_list_math/_list_zip.py` contains the Python registration and fallback path for
  the `list.zip` helper.
- `polars_list_math/_list_similarity.py` contains the Python API, Polars
  registration, scalar `py_list_similarity`, and fallback path for weighted list
  similarity.
- `polars_list_math/_json_object_items.py` and `_json_array_values.py` contain
  the Python APIs, `Expr.str` registrations, and fallback paths for converting
  JSON object and array strings to nested Polars values.
- `polars_list_math/_url_query_encode.py` contains the Python API and fallback
  path for encoding Struct or key/value-list expressions as URL query strings.
- `polars_list_math/_url_build.py` contains the Python API and fallback path for
  assembling URLs from optional component expressions.
- `rust/` is the standalone Rust crate used by maturin.
- `rust/src/lib.rs` is the Rust crate aggregator. Keep it small: register modules,
  configure the allocator, and expose the internal `_native` Python module.
- `rust/src/list_zip.rs` contains the Rust implementation of the `list_zip`
  Polars expression plugin.
- `rust/src/list_combinations.rs` contains the Rust implementation of the
  list pair-combination Polars expression plugins.
- `rust/src/list_similarity.rs` contains the Rust implementation of the
  `list_similarity` Polars expression plugin.
- `rust/src/list_mean_similarity.rs` contains the Rust implementation of the
  list-list mean similarity expression plugins.
- `rust/src/list_similarity_core.rs` contains shared Rust weighted list
  similarity helpers.
- `rust/src/json_object_items.rs` and `json_array_values.rs` contain the native
  JSON conversion expression plugins.
- `rust/src/url_query_encode.rs` and `url_build.rs` contain the native URL query
  encoding and URL assembly expression plugins.
- `tests/` uses pytest.
- `examples/` contains standalone runnable examples, with an index in
  `examples/README.md`.

When adding a new Rust-backed helper, add a new file under `rust/src/` and wire
it from `rust/src/lib.rs`. Keep each helper in its own Rust source file.

## Commands

Use the Makefile targets. They are the project contract.

```bash
make init
make install
make lock
make develop
make format
make lint
make test
make build
make package
```

Important checks before finishing a code change:

```bash
make lint
make test
```

For changes that touch packaging or Rust crate layout, also run:

```bash
make develop
make build
make package
```

## Tooling

- Use `uv` through the Makefile variable `UV_CMD`.
- `make init` and `make install` install dependencies without building the
  current maturin project.
- Run `make develop` when the editable native extension needs to be rebuilt.
- Rust is built through `maturin` using `rust/Cargo.toml`.
- The Python package imports the compiled plugin as `polars_list_math._native`.
- Importing `polars_list_math` registers helpers on both `Expr.list` and
  `Expr.str`; non-namespace helpers such as `url_query_encode` and `url_build`
  are exported from the package root.
- All expression helpers also have typed top-level exports. Treat these as the
  static-analysis API because Pyright/Pylance cannot augment Polars' namespace
  class declarations with runtime monkeypatches.
- `pyright` is configured in basic mode.
- `ruff` owns Python formatting and linting.
- Run `cargo fmt --manifest-path rust/Cargo.toml` after editing Rust files.

## Git And Artifacts

Track source files and lock files:

- `uv.lock`
- `rust/Cargo.lock`

Do not commit generated artifacts:

- `.venv/`
- `.pytest_cache/`
- `.ruff_cache/`
- `dist/`
- `target/`
- `rust/target/`
- `polars_list_math/*.so`
- `__pycache__/`

## Style Notes

- Prefer small, focused changes.
- Preserve the existing Python/Rust split.
- Keep Python registration/fallback code, Rust implementation, tests,
  documentation, and examples in feature-specific files.
- Do not reintroduce a root-level Rust crate; the Rust project lives in `rust/`.
- Prefer pytest-style tests and fixtures.
- Keep native and Python fallback behavior aligned, and test both paths for new
  expression helpers.
- Keep `list.zip` as one feature within `polars-list-math`, not the entire project
  identity.

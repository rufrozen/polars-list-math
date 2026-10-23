# Changelog

All notable changes to `polars-list-math` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-10-24

### Added

- Added `list_kendall_tau`, `list_mean_kendall_tau`,
  `list_mean_kendall_tau_to`, and `py_list_kendall_tau`, with equivalent
  `Expr.list` methods for the expression helpers.
- Compare the first-occurrence order of common IDs with an O(n log n) Rust
  implementation and Python fallback. Support primitive IDs, List/Array inputs,
  scalar broadcasting, and nanosecond temporal precision.
- Add optional `normalize=True` to map tau from `[-1, 1]` to `[0, 1]`.
  Undefined comparisons return null and are excluded from nested-list means.
- Added documentation, a runnable example, correctness tests, and six native
  performance scenarios showing 7.2–13.4x speedups in the recorded release build.

## [0.8.0] - 2026-10-17

### Added

- Added `nested_join` and `nested_lookup` with native Rust execution and Python
  fallbacks for matching scalar, ancestor and leaf keys without exploding left.
- Preserve list structure and row/item order, enforce exact key types and
  right-key uniqueness, and optionally skip null keys without removing positions.
- Support all DataFrame/LazyFrame combinations, multiple typed payload columns,
  and public expression lookup against shared or row-local dictionaries.
- Added correctness tests and a performance experiment comparing native,
  fallback, and explode/join/implode with precomputed positional indexes.

### Changed

- Optimize native nested joins with reusable traversal buffers, cached ancestor
  keys, a compact right index with exact collision checks, and bulk Arrow
  payload gathering. Keep full-right validation and null/empty list semantics.

## [0.7.0] - 2026-09-19

### Added

- Added `find_null_schema` to discover columns and fields without non-null
  values across a LazyFrame, including nested List, Array, and Struct values.
- Added `clean_schema` to remove a supplied subschema, including null-schema
  results, and recursively prune containers whose fields have all been removed.
- Added `apply_schema` to select and cast a LazyFrame to the resulting schema
  while preserving nested values and parent nulls.
- Added a guide, runnable example, and 79 schema-helper test cases covering
  mixed nesting, empty data, dtype metadata, validation, and lazy execution.

### Changed

- Clarified the library's focus on DataFrames representing nested struct fields
  as separate list columns, with a diagram of the representation.
- Combined the README API tables around top-level functions, with an optional
  column for equivalent namespace methods.

## [0.6.0] - 2026-09-18

### Added

- Added `ZipBuilder` for nested list zips with independent zip options and
  ordered eval/filter operations at each depth; `build()` returns a Polars expression.
- Added optional scalar broadcasting to `list_zip` and `Expr.list.zip`, including
  support for null scalars and independent padding of list inputs.
- Added directed SERP containment (`|A ∩ B| / |A|`) with pairwise, mean-within-group,
  and mean-to-reference expression helpers, native Rust implementations, and
  Python fallbacks. Added `py_list_containment` for Python sequences.
- Added containment documentation, runnable examples, and native-versus-fallback
  performance experiments.

## [0.5.0] - 2026-09-17

### Removed

- Removed `polars_list_math.typed_polars` and its tests, examples, guide, and
  construction benchmark. Typed models and schemas are outside this package's
  scope and may be developed separately.

## Release status notice

Versions `0.4.0`, `0.4.1`, and `0.4.2` were unsuccessful preview releases of
the typed Polars API. Their model, schema, and projection interfaces changed
rapidly. They are not recommended for use. Version `0.4.3` replaced those
interfaces; version `0.5.0` removes `polars_list_math.typed_polars`
entirely. The release notes below describe historical APIs.

The list, similarity, game, JSON, and URL expression helpers remain available
in `0.5.0`.

## [0.4.3] - 2026-08-22

### Added

- Added the tuple-oriented `typed_polars` model system for schema-bound
  dataclasses and typed `NamedTuple` records.
- Added explicit `Schema`, `Builder`, and `Context` APIs, including nested
  structs, list structs, flat structs, `FlatDict`, and runtime-bound
  `FlatTuple` fields.
- Added `dfs_to_lazy_df` for exposing DataFrame batches as a pushdown-aware
  lazy source.
- Added construction benchmarks and native-versus-fallback performance tests
  with baseline and complex workloads.

### Changed

- Replaced the experimental `0.4.0`–`0.4.2` typed model/projection design with
  an explicit schema compiler and immutable physical serialization plans.
- Root models now cache their compiled builder; `Schema` itself is stateless.
- Model/schema binding is permissive by default and supports an explicit
  recursive strict mode.
- Expanded typed Polars validation, deserialization edge-case coverage,
  documentation, and runnable examples.

### Upgrade notes

- Treat the `typed_polars` API in this release as a replacement for the APIs
  shipped in `0.4.0`, `0.4.1`, and `0.4.2`, not as a drop-in patch.
- Use `@model(schema=...)` on root records and use the model's cached `Builder`
  for conversion. The guide is available in the `0.4.3` source distribution.

## [0.4.2] - 2026-08-17

Unsuccessful preview release; superseded by `0.4.3` and not recommended for
use. It added typed scalar overrides and partial typed-frame improvements to an
API that was subsequently replaced.

## [0.4.1] - 2026-08-17

Unsuccessful preview release; superseded by `0.4.3` and not recommended for
use. It revised the initial typed Polars schema and projection model, but the
design remained unstable.

## [0.4.0] - 2026-08-16

Unsuccessful preview release; superseded by `0.4.3` and not recommended for
use. It introduced the first experimental typed Polars schemas, models, and
projections.

## [0.3.0] - 2026-08-12

### Added

- Added the native and Python fallback implementations of
  `expected_value_of_game`.

## [0.2.0] - 2026-08-04

### Added

- Added typed top-level expression helpers for static-analysis tools while
  retaining the dynamically registered Polars namespace methods.

## [0.1.2] - 2026-08-04

### Added

- Added JSON object and array conversion helpers.
- Added URL query encoding and URL construction helpers.
- Added `skip_self` support to list combinations.

## [0.1.1] - 2026-07-23

### Fixed

- Corrected the local distribution-check workflow so dependencies can be
  installed without prematurely building the current maturin project.

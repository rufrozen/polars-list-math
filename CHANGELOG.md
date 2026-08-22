# Changelog

All notable changes to `polars-list-math` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Release status notice

Versions `0.4.0`, `0.4.1`, and `0.4.2` were unsuccessful preview releases of
the typed Polars API. Their model, schema, and projection interfaces changed
rapidly and do not represent the supported design. They are not recommended
for use. Upgrade directly to `0.4.3`; migration may require rewriting code that
used `polars_list_math.typed_polars` from those releases.

The list, similarity, game, JSON, and URL expression helpers remain available
in `0.4.3`.

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
  for conversion. See `docs/typed_polars.md` for the current design.

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

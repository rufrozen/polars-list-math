# typed_polars

Nested models are ordinary dataclasses or typed `NamedTuple` records. The root
record uses `@tp.model(schema=...)`, which validates the complete model tree and
caches its physical serialization plan in a `Builder` attached to the root
model. `Schema` is stateless; its conversion methods resolve the builder from
the explicitly supplied model or row.

Model/schema binding is non-strict by default: conversion uses matching fields,
ignores schema-only fields, and requires defaults for model-only fields. Pass
`strict=True` to `@tp.model` to require identical field sets recursively.

## Internal layers

The implementation has an acyclic module graph and no function-local imports:

- `_plans.py` contains immutable logical and physical cache structures;
- `_records.py` normalizes dataclass and typed NamedTuple fields;
- `context.py` stores runtime `FlatDict` and `FlatTuple` key bindings;
- `_binding.py` resolves builders cached directly on root model classes;
- `schema.py` contains the schema DSL and resolves builders through that helper;
- `_compiler.py` validates record trees and compiles physical plans;
- `_codec.py` performs runtime serialization and deserialization;
- `builder.py` composes compiler, codec, schema, and model binding;
- `model.py` contains only the required root-model decorator.

Use `dataclasses.field` for defaults and factories. There is deliberately no
model-level `field`, dtype override, alias, or untyped extras API. Runtime
dynamic columns use `FlatDict[T]` or `FlatTuple[T]` plus explicit
`Context.bind()` keys. Flat Struct storage is configured by `FlatStruct` or
`FlatListStruct` in the schema; their `divider` defaults to `_`.
See [`docs/typed_polars.md`](../../docs/typed_polars.md) and
the regular and flat examples in [`examples`](../../examples/README.md).

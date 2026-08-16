"""Column-oriented DataFrame builders for typed Polars schemas."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, cast

import polars as pl

from .model import (
    ListStruct,
    Schema,
    Struct,
    _require_schema_type,
    _serialize_value,
)


def build_frame(
    schema_cls: type[Schema],
    rows: list[Schema],
    *,
    strict: bool,
    extra_schema: Mapping[str, Any] | None,
) -> pl.DataFrame:
    schema = _resolved_schema(
        schema_cls,
        rows,
        strict=strict,
        extra_schema=extra_schema,
    )
    field_buffers = [(plan, []) for plan in schema_cls.__field_plan__]

    for row in rows:
        for plan, buffer in field_buffers:
            buffer.append(plan.serialize_alias(row._values[plan.index]))

    series: list[pl.Series] = [
        pl.Series(plan.alias, buffer, dtype=schema[plan.alias], strict=strict)
        for plan, buffer in field_buffers
    ]
    extras_field = schema_cls.__extras_field__
    if extras_field is not None:
        declared = {plan.alias for plan in schema_cls.__field_plan__}
        for name, dtype in schema.items():
            if name in declared:
                continue
            values = [
                _serialize_value(
                    getattr(row, extras_field.name).get(name),
                    by_alias=True,
                )
                for row in rows
            ]
            series.append(pl.Series(name, values, dtype=dtype, strict=strict))
    return pl.DataFrame(series)


def build_flat_frame(
    schema_cls: type[Schema],
    rows: list[Schema],
    *,
    strict: bool,
    extra_schema: Mapping[str, Any] | None,
) -> pl.DataFrame:
    nested_schema = _resolved_schema(
        schema_cls,
        rows,
        strict=strict,
        extra_schema=extra_schema,
    )
    fields = _flattened_fields(schema_cls, nested_schema)
    plans = {plan.alias: plan for plan in schema_cls.__field_plan__}
    extras_field = schema_cls.__extras_field__
    buffers = [[] for _ in fields]
    roots = _build_path_trees(fields)

    for row in rows:
        extras = getattr(row, extras_field.name) if extras_field is not None else None
        for root, tree in roots.items():
            plan = plans.get(root)
            if plan is not None:
                root_value = row._values[plan.index]
            else:
                if extras is None:
                    raise AssertionError(f"Missing Extras descriptor for {root!r}")
                root_value = extras.get(root)
            _append_tree(tree, root_value, buffers)

    return pl.DataFrame(
        [
            pl.Series(field.name, buffer, dtype=field.dtype, strict=strict)
            for field, buffer in zip(fields, buffers, strict=True)
        ]
    )


def flatten_schema(
    schema_cls: type[Schema],
    schema: Mapping[str, Any],
) -> pl.Schema:
    return pl.Schema({field.name: field.dtype for field in _flattened_fields(schema_cls, schema)})


def _resolved_schema(
    schema_cls: type[Schema],
    rows: list[Schema],
    *,
    strict: bool,
    extra_schema: Mapping[str, Any] | None,
) -> pl.Schema:
    """Resolve declared and dynamic fields without constructing row dictionaries."""
    if extra_schema is None and not _has_extras(schema_cls):
        return schema_cls.polars_schema()

    explicit = extra_schema or {}
    base = schema_cls.polars_schema(explicit)
    result: dict[str, Any] = {}

    for plan in schema_cls.__field_plan__:
        info = plan.info
        nested_explicit = explicit.get(plan.alias)
        values = [row._values[plan.index] for row in rows]
        if isinstance(info, Struct):
            nested = _require_schema_type(info.python_type, info)
            nested_rows = [value for value in values if isinstance(value, Schema)]
            result[plan.alias] = pl.Struct(
                _resolved_schema(
                    nested,
                    nested_rows,
                    strict=strict,
                    extra_schema=cast(Mapping[str, Any] | None, nested_explicit),
                )
            )
        elif isinstance(info, ListStruct):
            nested = _require_schema_type(info.python_type, info)
            nested_rows = [
                item
                for value in values
                if isinstance(value, list)
                for item in value
                if isinstance(item, Schema)
            ]
            result[plan.alias] = pl.List(
                pl.Struct(
                    _resolved_schema(
                        nested,
                        nested_rows,
                        strict=strict,
                        extra_schema=cast(
                            Mapping[str, Any] | None,
                            nested_explicit,
                        ),
                    )
                )
            )
        else:
            result[plan.alias] = base[plan.alias]

    extras_field = schema_cls.__extras_field__
    if extras_field is None:
        return pl.Schema(result)

    extra_values: dict[str, list[Any]] = {name: [] for name in explicit if name not in result}
    for row in rows:
        extras = getattr(row, extras_field.name)
        conflicts = set(result) & set(extras)
        if conflicts:
            raise TypeError(
                f"Extras conflict with declared field(s): {', '.join(sorted(conflicts))}"
            )
        for name in extras:
            extra_values.setdefault(name, [])

    for name, values in extra_values.items():
        dtype = base.get(name)
        if dtype is None:
            values.extend(
                _serialize_value(
                    getattr(row, extras_field.name).get(name),
                    by_alias=True,
                )
                for row in rows
            )
            dtype = pl.Series(name, values, strict=strict).dtype
        result[name] = dtype

    return pl.Schema(result)


@cache
def _has_extras(schema_cls: type[Schema]) -> bool:
    if schema_cls.__extras_field__ is not None:
        return True
    for plan in schema_cls.__field_plan__:
        if isinstance(plan.info, (Struct, ListStruct)):
            nested = _require_schema_type(plan.info.python_type, plan.info)
            if _has_extras(nested):
                return True
    return False


@dataclass(frozen=True, slots=True)
class _FlatField:
    name: str
    dtype: Any
    root: str
    steps: tuple[tuple[str, str], ...]


def _flattened_fields(
    schema_cls: type[Schema],
    schema: Mapping[str, Any],
) -> list[_FlatField]:
    result: list[_FlatField] = []
    for root, dtype in schema.items():
        plan = schema_cls.__field_plan_by_alias__.get(root)
        info = plan.info if plan is not None else None
        name = info.flat_alias if info is not None and info.flat_alias else root
        nested = _nested_schema(info)
        result.extend(
            _flat_fields(
                name,
                dtype,
                root=root,
                steps=(),
                schema_cls=nested,
            )
        )
    return result


def _flat_fields(
    name: str,
    dtype: Any,
    *,
    root: str,
    steps: tuple[tuple[str, str], ...],
    schema_cls: type[Schema] | None,
) -> list[_FlatField]:
    if isinstance(dtype, pl.Struct):
        result: list[_FlatField] = []
        for child in dtype.fields:
            plan = (
                schema_cls.__field_plan_by_alias__.get(child.name)
                if schema_cls is not None
                else None
            )
            info = plan.info if plan is not None else None
            child_name = info.flat_alias if info is not None and info.flat_alias else child.name
            result.extend(
                _flat_fields(
                    f"{name}:{child_name}",
                    child.dtype,
                    root=root,
                    steps=(*steps, ("struct", child.name)),
                    schema_cls=_nested_schema(info),
                )
            )
        return result

    if isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct):
        return [
            _FlatField(
                name=child.name,
                dtype=pl.List(child.dtype),
                root=child.root,
                steps=child.steps,
            )
            for child in _flat_fields(
                name,
                dtype.inner,
                root=root,
                steps=(*steps, ("list", "")),
                schema_cls=schema_cls,
            )
        ]

    return [_FlatField(name=name, dtype=dtype, root=root, steps=steps)]


def _nested_schema(info: Any) -> type[Schema] | None:
    if isinstance(info, (Struct, ListStruct)):
        return _require_schema_type(info.python_type, info)
    return None


@dataclass(slots=True)
class _PathTree:
    leaf: int | None = None
    struct: dict[str, _PathTree] | None = None
    list_item: _PathTree | None = None
    leaves: tuple[int, ...] = ()


def _build_path_trees(fields: list[_FlatField]) -> dict[str, _PathTree]:
    roots: dict[str, _PathTree] = {}
    for index, field in enumerate(fields):
        node = roots.setdefault(field.root, _PathTree())
        for kind, name in field.steps:
            if kind == "struct":
                if node.struct is None:
                    node.struct = {}
                node = node.struct.setdefault(name, _PathTree())
            elif kind == "list":
                if node.list_item is None:
                    node.list_item = _PathTree()
                node = node.list_item
            else:
                raise AssertionError(f"Unknown flat serialization step: {kind}")
        node.leaf = index

    for root in roots.values():
        _cache_leaves(root)
    return roots


def _cache_leaves(tree: _PathTree) -> tuple[int, ...]:
    if tree.leaf is not None:
        tree.leaves = (tree.leaf,)
    elif tree.list_item is not None:
        tree.leaves = _cache_leaves(tree.list_item)
    elif tree.struct is not None:
        tree.leaves = tuple(leaf for child in tree.struct.values() for leaf in _cache_leaves(child))
    else:
        raise AssertionError("Flat path tree has no leaf")
    return tree.leaves


def _append_tree(
    tree: _PathTree,
    value: Any,
    buffers: list[list[Any]] | dict[int, list[Any]],
) -> None:
    if value is None:
        for leaf in tree.leaves:
            buffers[leaf].append(None)
        return

    if tree.leaf is not None:
        buffers[tree.leaf].append(value)
        return

    if tree.list_item is not None:
        if isinstance(value, Mapping):
            value = [{"key": key, "value": item} for key, item in value.items()]
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")
        item_buffers = {leaf: [] for leaf in tree.leaves}
        for item in value:
            _append_tree(tree.list_item, item, item_buffers)
        for leaf in tree.leaves:
            buffers[leaf].append(item_buffers[leaf])
        return

    if tree.struct is None:
        raise AssertionError("Flat path tree has no children")
    for name, child in tree.struct.items():
        _append_tree(child, _struct_value(value, name), buffers)


def _struct_value(value: Any, name: str) -> Any:
    if isinstance(value, Schema):
        plan = type(value).__field_plan_by_alias__.get(name)
        if plan is not None:
            return value._values[plan.index]
        extras_field = type(value).__extras_field__
        if extras_field is not None:
            return getattr(value, extras_field.name).get(name)
    elif isinstance(value, Mapping):
        return value.get(name)
    raise TypeError(f"Expected Schema or mapping, got {type(value).__name__}")

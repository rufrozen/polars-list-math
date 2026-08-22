"""Runtime serialization and deserialization for compiled model plans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import polars as pl

from ._binding import get_builder
from ._const import MODEL_PLAN_ATTRIBUTE, MODEL_SCHEMA_ATTRIBUTE
from ._plans import PhysicalPlan
from .context import Context, context_keys
from .schema import FlatDict, FlatTuple, ListStruct, Schema, Struct


def to_frame_many(
    cls: type[Any],
    rows: Iterable[Any],
    *,
    context: Context | None = None,
    strict: bool = True,
) -> pl.DataFrame:
    materialized = list(rows)
    plan = physical_plan(cls, context=context)
    for row in materialized:
        validate_row(cls, row, context=context)
    physical_rows = [plan.serialize(row) for row in materialized]
    return pl.DataFrame(physical_rows, orient="row", schema=plan.schema, strict=strict)


def iter_frame(
    cls: type[Any], frame: pl.DataFrame, *, strict_schema: bool = False
) -> Iterable[Any]:
    if strict_schema:
        assert_frame_schema(cls, frame)
    for row in frame.iter_rows(named=True):
        yield from_dict(cls, row, by_polars_name=True, ignore_unknown=not strict_schema)


def from_frame(cls: type[Any], frame: pl.DataFrame, *, strict_schema: bool = False) -> Any:
    if frame.height != 1:
        raise ValueError(f"Expected exactly one row, got {frame.height}")
    return next(iter(iter_frame(cls, frame, strict_schema=strict_schema)))


def assert_frame_schema(cls: type[Any], frame: pl.DataFrame) -> None:
    context = infer_context(cls, frame.schema)
    expected = physical_plan(cls, context=context).schema
    if frame.schema != expected:
        raise TypeError(f"Unexpected DataFrame schema: expected {expected}, got {frame.schema}")


def to_dict(
    value: Any,
    *,
    by_polars_name: bool = False,
    context: Context | None = None,
    _context_path: tuple[object, ...] = (),
) -> dict[str, Any]:
    cls = type(value)
    plan = getattr(cls, MODEL_PLAN_ATTRIBUTE)
    schema = require_schema(cls) if by_polars_name else None
    result: dict[str, Any] = {}
    for item in plan.fields:
        child = getattr(value, item.name)
        column = cast(type[Schema], schema).fields().get(item.name) if by_polars_name else None
        if by_polars_name and column is None:
            continue
        if by_polars_name and item.kind == "flat_dict":
            assert isinstance(column, FlatDict)
            field_path = _context_path + (column,)
            _validate_flat_dict(
                item.name,
                child,
                column,
                context,
                field_path,
            )
            for dynamic_key in context_keys(context, column, field_path):
                result[column.physical_name(dynamic_key)] = (
                    None if child is None else child.get(dynamic_key)
                )
            continue
        if by_polars_name and item.kind == "flat_tuple":
            assert isinstance(column, FlatTuple)
            field_path = _context_path + (column,)
            keys = context_keys(context, column, field_path)
            _validate_flat_tuple(item.name, child, keys)
            for index, dynamic_key in enumerate(keys):
                result[column.physical_name(dynamic_key)] = None if child is None else child[index]
            continue
        key = polars_name(cls, item.name) if by_polars_name else item.name
        nested_context_path = _context_path + (column,) if column is not None else _context_path
        if item.kind == "struct" and child is not None:
            child = to_dict(
                child,
                by_polars_name=by_polars_name,
                context=context,
                _context_path=nested_context_path,
            )
        elif item.kind == "list_struct" and child is not None:
            child = [
                to_dict(
                    nested,
                    by_polars_name=by_polars_name,
                    context=context,
                    _context_path=nested_context_path,
                )
                for nested in child
            ]
        result[key] = child
    return result


def from_dict(
    cls: type[Any],
    data: Mapping[str, Any],
    *,
    by_polars_name: bool,
    ignore_unknown: bool,
) -> Any:
    plan = getattr(cls, MODEL_PLAN_ATTRIBUTE)
    schema = require_schema(cls)
    remaining = unflatten(cls, schema, data) if by_polars_name else dict(data)
    kwargs: dict[str, Any] = {}
    for item in plan.fields:
        if by_polars_name and item.name not in schema.fields():
            continue
        key = polars_name(cls, item.name) if by_polars_name else item.name
        if key not in remaining:
            continue
        value = remaining.pop(key)
        if item.kind == "struct" and value is not None:
            assert item.nested is not None
            if not isinstance(value, Mapping):
                raise TypeError(f"Field {item.name!r} must be a mapping")
            value = from_dict(
                item.nested,
                value,
                by_polars_name=by_polars_name,
                ignore_unknown=ignore_unknown,
            )
        elif item.kind == "list_struct" and value is not None:
            assert item.nested is not None
            if not isinstance(value, list) or not all(
                isinstance(child, Mapping) for child in value
            ):
                raise TypeError(f"Field {item.name!r} must be a list of mappings")
            value = [
                from_dict(
                    item.nested,
                    child,
                    by_polars_name=by_polars_name,
                    ignore_unknown=ignore_unknown,
                )
                for child in value
            ]
        elif item.kind == "dict" and isinstance(value, list):
            value = {entry["key"]: entry["value"] for entry in value}
        kwargs[item.name] = value
    if remaining and not ignore_unknown:
        raise TypeError(f"Unexpected key(s) for {cls.__name__}: {', '.join(sorted(remaining))}")
    return cls(**kwargs)


def unflatten(cls: type[Any], schema: type[Schema], data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    for item in getattr(cls, MODEL_PLAN_ATTRIBUTE).fields:
        column = schema.fields().get(item.name)
        if column is None:
            continue
        if isinstance(column, FlatDict):
            prefix = f"{column.polars_name}{column.divider}"
            selected: dict[str, Any] = {}
            for key in tuple(result):
                if not key.startswith(prefix):
                    continue
                value = result.pop(key)
                if value is not None:
                    selected[key[len(prefix) :]] = value
            result[column.polars_name] = selected
            continue
        if isinstance(column, FlatTuple):
            prefix = f"{column.polars_name}{column.divider}"
            selected_values: list[Any] = []
            for key in tuple(result):
                if key.startswith(prefix):
                    selected_values.append(result.pop(key))
            result[column.polars_name] = tuple(selected_values)
            continue
        if item.kind not in ("struct", "list_struct") or not cast(Any, column).flat:
            continue
        prefix = f"{column.polars_name}{cast(Any, column).divider}"
        selected = {
            key[len(prefix) :]: result.pop(key) for key in tuple(result) if key.startswith(prefix)
        }
        if item.kind == "struct":
            result[column.polars_name] = (
                None if selected and all(value is None for value in selected.values()) else selected
            )
            continue
        lengths = {len(value) for value in selected.values() if value is not None}
        if len(lengths) > 1:
            raise TypeError(
                f"Flat ListStruct columns for {column.polars_name!r} have different lengths"
            )
        result[column.polars_name] = (
            None
            if not lengths
            else [
                {name: value[index] for name, value in selected.items()}
                for index in range(next(iter(lengths)))
            ]
        )
    return result


def require_schema(cls: type[Any]) -> type[Schema]:
    schema = getattr(cls, MODEL_SCHEMA_ATTRIBUTE)
    if schema is None:
        raise TypeError(
            f"{cls.__name__} has no schema; DataFrame serialization requires @model(schema=...)"
        )
    return schema


def polars_name(cls: type[Any], field_name: str) -> str:
    return require_schema(cls).fields()[field_name].polars_name


def physical_plan(
    cls: type[Any],
    *,
    context: Context | None = None,
) -> PhysicalPlan:
    schema = require_schema(cls)
    return cast(
        PhysicalPlan,
        get_builder(cls, schema=schema).physical_plan_for(context),
    )


def validate_row(
    cls: type[Any],
    row: Any,
    *,
    context: Context | None,
    _context_path: tuple[object, ...] = (),
) -> None:
    if type(row) is not cls:
        raise TypeError(f"Expected {cls.__name__}, got {type(row).__name__}")
    schema = require_schema(cls)
    for item in getattr(cls, MODEL_PLAN_ATTRIBUTE).fields:
        if item.kind == "flat_dict":
            column = schema.fields()[item.name]
            assert isinstance(column, FlatDict)
            field_path = _context_path + (column,)
            _validate_flat_dict(
                item.name,
                getattr(row, item.name),
                column,
                context,
                field_path,
            )
            continue
        if item.kind == "flat_tuple":
            column = schema.fields()[item.name]
            assert isinstance(column, FlatTuple)
            _validate_flat_tuple(
                item.name,
                getattr(row, item.name),
                context_keys(context, column, _context_path + (column,)),
            )
            continue
        if item.nested is None:
            continue
        if item.name not in schema.fields():
            continue
        column = schema.fields()[item.name]
        value = getattr(row, item.name)
        if value is None:
            continue
        if item.kind == "struct":
            validate_row(
                item.nested,
                value,
                context=context,
                _context_path=_context_path + (column,),
            )
            continue
        if not isinstance(value, list):
            raise TypeError(f"Field {item.name!r} must be a list of {item.nested.__name__}")
        for child in value:
            validate_row(
                item.nested,
                child,
                context=context,
                _context_path=_context_path + (column,),
            )


def _validate_flat_dict(
    name: str,
    value: Any,
    column: FlatDict[Any],
    context: Context | None,
    context_path: tuple[object, ...],
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"Field {name!r} must be a mapping")
    allowed = set(context_keys(context, column, context_path))
    unknown = set(value) - allowed
    if unknown:
        rendered = ", ".join(sorted(repr(key) for key in unknown))
        raise TypeError(f"FlatDict field {name!r} has unbound key(s): {rendered}")


def _validate_flat_tuple(name: str, value: Any, keys: tuple[str, ...]) -> None:
    if value is None:
        return
    if not isinstance(value, tuple):
        raise TypeError(f"FlatTuple field {name!r} must be a tuple")
    if len(value) != len(keys):
        raise TypeError(
            f"FlatTuple field {name!r} has {len(value)} value(s), "
            f"but its context binds {len(keys)} position(s)"
        )


def infer_context(cls: type[Any], physical_schema: Mapping[str, Any]) -> Context:
    """Infer dynamic field keys from a physical schema for reverse conversion."""
    context = Context()
    _infer_context(cls, require_schema(cls), physical_schema, context, ())
    return context


def _infer_context(
    cls: type[Any],
    schema: type[Schema],
    physical_fields: Mapping[str, Any],
    context: Context,
    context_path: tuple[object, ...],
) -> None:
    for item in getattr(cls, MODEL_PLAN_ATTRIBUTE).fields:
        column = schema.fields().get(item.name)
        if column is None:
            continue
        if isinstance(column, (FlatDict, FlatTuple)):
            prefix = f"{column.polars_name}{column.divider}"
            keys = [name[len(prefix) :] for name in physical_fields if name.startswith(prefix)]
            context._bind_path(column, context_path + (column,), keys)
            continue
        if item.nested is None or not isinstance(column, (Struct, ListStruct)):
            continue
        if column.flat:
            prefix = f"{column.polars_name}{column.divider}"
            nested_fields: dict[str, Any] = {}
            for name, dtype in physical_fields.items():
                if not name.startswith(prefix):
                    continue
                nested_dtype = dtype
                if isinstance(column, ListStruct) and isinstance(dtype, pl.List):
                    nested_dtype = dtype.inner
                nested_fields[name[len(prefix) :]] = nested_dtype
            _infer_context(
                item.nested,
                column.schema,
                nested_fields,
                context,
                context_path + (column,),
            )
            continue
        dtype = physical_fields.get(column.polars_name)
        if isinstance(column, ListStruct):
            dtype = dtype.inner if isinstance(dtype, pl.List) else None
        if isinstance(dtype, pl.Struct):
            _infer_context(
                item.nested,
                column.schema,
                dtype.to_schema(),
                context,
                context_path + (column,),
            )

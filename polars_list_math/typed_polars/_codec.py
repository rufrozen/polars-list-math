"""Runtime serialization and deserialization for compiled model plans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import polars as pl

from ._binding import get_builder
from ._plans import PhysicalPlan
from .schema import Schema


def to_frame_many(cls: type[Any], rows: Iterable[Any], *, strict: bool = True) -> pl.DataFrame:
    materialized = list(rows)
    plan = physical_plan(cls)
    for row in materialized:
        validate_row(cls, row)
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
    expected = physical_plan(cls).schema
    if frame.schema != expected:
        raise TypeError(f"Unexpected DataFrame schema: expected {expected}, got {frame.schema}")


def to_dict(value: Any, *, by_polars_name: bool = False) -> dict[str, Any]:
    cls = type(value)
    plan = cls.__tp2_plan__
    schema = require_schema(cls) if by_polars_name else None
    result: dict[str, Any] = {}
    for item in plan.fields:
        child = getattr(value, item.name)
        if by_polars_name and item.name not in cast(type[Schema], schema).fields():
            continue
        key = polars_name(cls, item.name) if by_polars_name else item.name
        if item.kind == "struct" and child is not None:
            child = to_dict(child, by_polars_name=by_polars_name)
        elif item.kind == "list_struct" and child is not None:
            child = [to_dict(nested, by_polars_name=by_polars_name) for nested in child]
        result[key] = child
    return result


def from_dict(
    cls: type[Any],
    data: Mapping[str, Any],
    *,
    by_polars_name: bool,
    ignore_unknown: bool,
) -> Any:
    plan = cls.__tp2_plan__
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
    for item in cls.__tp2_plan__.fields:
        column = schema.fields().get(item.name)
        if column is None:
            continue
        if item.kind not in ("struct", "list_struct") or not cast(Any, column).flat:
            continue
        prefix = f"{column.polars_name}{cast(Any, column).flat_divider}"
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
    schema = cls.__tp_schema__
    if schema is None:
        raise TypeError(
            f"{cls.__name__} has no schema; DataFrame serialization requires @model(schema=...)"
        )
    return schema


def polars_name(cls: type[Any], field_name: str) -> str:
    return require_schema(cls).fields()[field_name].polars_name


def physical_plan(cls: type[Any]) -> PhysicalPlan:
    schema = require_schema(cls)
    return cast(PhysicalPlan, get_builder(cls, schema=schema).physical_plan)


def validate_row(cls: type[Any], row: Any) -> None:
    if type(row) is not cls:
        raise TypeError(f"Expected {cls.__name__}, got {type(row).__name__}")
    schema = require_schema(cls)
    for item in cls.__tp2_plan__.fields:
        if item.name not in schema.fields() or item.nested is None:
            continue
        value = getattr(row, item.name)
        if value is None:
            continue
        if item.kind == "struct":
            validate_row(item.nested, value)
            continue
        if not isinstance(value, list):
            raise TypeError(f"Field {item.name!r} must be a list of {item.nested.__name__}")
        for child in value:
            validate_row(item.nested, child)

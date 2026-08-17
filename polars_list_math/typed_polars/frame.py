"""DataFrame storage for typed schemas with fixed hybrid Struct layouts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import polars as pl

from .dtypes import is_nullable
from .model import ListStruct, Schema, Struct, _require_schema_type, _serialize_value


def build_frame(
    schema_cls: type[Schema],
    rows: list[Schema],
    *,
    strict: bool,
    extra_schema: Mapping[str, Any] | None,
) -> pl.DataFrame:
    logical = _resolved_schema(schema_cls, rows, strict=strict, extra_schema=extra_schema)
    schema = storage_schema(schema_cls, logical)
    buffers = {name: [] for name in schema}
    for row in rows:
        values = _storage_mapping(schema_cls, row, logical, strict=strict)
        for name, buffer in buffers.items():
            buffer.append(values.get(name))
    return pl.DataFrame(
        [
            pl.Series(name, values, dtype=schema[name], strict=strict)
            for name, values in buffers.items()
        ]
    )


def storage_schema(
    schema_cls: type[Schema], logical_schema: Mapping[str, Any] | None = None
) -> pl.Schema:
    """Compile the single physical schema declared by all Struct fields."""
    logical = logical_schema or _logical_schema(schema_cls)
    result: dict[str, Any] = {}
    for name, dtype in logical.items():
        plan = schema_cls.__field_plan_by_alias__.get(name)
        info = plan.info if plan is not None else None
        if isinstance(info, Struct):
            nested = _require_schema_type(info.python_type, info)
            children = storage_schema(nested, _struct_fields(dtype))
            if info.flat:
                _extend_flat_schema(result, name, info.flat_divider, children, 0)
            else:
                _put_schema(result, name, pl.Struct(children))
        elif isinstance(info, ListStruct):
            nested = _require_schema_type(info.python_type, info)
            children = storage_schema(nested, _struct_fields(_list_inner(dtype)))
            if info.flat:
                _extend_flat_schema(result, name, info.flat_divider, children, 1)
            else:
                _put_schema(result, name, pl.List(pl.Struct(children)))
        else:
            _put_schema(result, name, dtype)
    return pl.Schema(result)


def deserialize_row(schema_cls: type[Schema], data: Mapping[str, Any]) -> Schema:
    return schema_cls.from_dict(
        _logical_mapping(schema_cls, data), by_alias=True, forbid_extra=False
    )


def _logical_schema(schema_cls: type[Schema]) -> pl.Schema:
    return pl.Schema({plan.alias: plan.info.polars_dtype for plan in schema_cls.__field_plan__})


def _extend_flat_schema(
    result: dict[str, Any], prefix: str, divider: str, nested: Mapping[str, Any], list_depth: int
) -> None:
    for child, dtype in nested.items():
        name = f"{prefix}{divider}{child}"
        for _ in range(list_depth):
            dtype = pl.List(dtype)
        _put_schema(result, name, dtype)


def _put_schema(result: dict[str, Any], name: str, dtype: Any) -> None:
    if name in result:
        raise TypeError(f"Physical column name conflicts with {name!r}")
    result[name] = dtype


def _storage_mapping(
    schema_cls: type[Schema],
    row: Schema,
    logical_schema: Mapping[str, Any],
    *,
    strict: bool,
    path: str = "",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    declared: set[str] = set()
    for plan in schema_cls.__field_plan__:
        info, value, dtype = plan.info, row._values[plan.index], logical_schema[plan.alias]
        field_path = f"{path}.{plan.name}" if path else plan.name
        if value is None and not is_nullable(info.storage_type):
            if strict:
                raise TypeError(
                    f"Field {field_path!r} does not accept None; declare its type with | None"
                )
            if info.required:
                raise TypeError(f"Field {field_path!r} does not accept None and has no default")
            value = info.get_default()
            if value is None:
                raise TypeError(f"Default for non-nullable field {field_path!r} must not be None")
        declared.add(plan.alias)
        if isinstance(info, Struct):
            nested_cls = _require_schema_type(info.python_type, info)
            nested_schema = _struct_fields(dtype)
            nested_values = (
                {name: None for name in storage_schema(nested_cls, nested_schema)}
                if value is None
                else _storage_mapping(
                    nested_cls, value, nested_schema, strict=strict, path=field_path
                )
            )
            if info.flat:
                result.update(
                    {
                        f"{plan.alias}{info.flat_divider}{name}": item
                        for name, item in nested_values.items()
                    }
                )
            else:
                result[plan.alias] = None if value is None else nested_values
        elif isinstance(info, ListStruct):
            nested_cls = _require_schema_type(info.python_type, info)
            nested_schema = _struct_fields(_list_inner(dtype))
            if info.flat:
                names = tuple(storage_schema(nested_cls, nested_schema))
                if value is None:
                    result.update(
                        {f"{plan.alias}{info.flat_divider}{name}": None for name in names}
                    )
                else:
                    items = [
                        _storage_mapping(
                            nested_cls,
                            item,
                            nested_schema,
                            strict=strict,
                            path=f"{field_path}[{index}]",
                        )
                        for index, item in enumerate(value)
                    ]
                    result.update(
                        {
                            f"{plan.alias}{info.flat_divider}{name}": [
                                item.get(name) for item in items
                            ]
                            for name in names
                        }
                    )
            else:
                result[plan.alias] = (
                    None
                    if value is None
                    else [
                        _storage_mapping(
                            nested_cls,
                            item,
                            nested_schema,
                            strict=strict,
                            path=f"{field_path}[{index}]",
                        )
                        for index, item in enumerate(value)
                    ]
                )
        else:
            result[plan.alias] = plan.serialize_alias(value)

    extras = schema_cls.__extras_field__
    if extras is not None:
        for name in logical_schema:
            if name not in declared:
                result[name] = _serialize_value(getattr(row, extras.name).get(name), by_alias=True)
    return result


def _logical_mapping(schema_cls: type[Schema], data: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    consumed: set[str] = set()
    for plan in schema_cls.__field_plan__:
        info = plan.info
        if isinstance(info, (Struct, ListStruct)) and info.flat:
            prefix = f"{plan.alias}{info.flat_divider}"
            columns = {
                key[len(prefix) :]: value for key, value in data.items() if key.startswith(prefix)
            }
            consumed.update(key for key in data if key.startswith(prefix))
            if not columns or all(value is None for value in columns.values()):
                result[plan.alias] = None
            elif isinstance(info, Struct):
                nested_cls = _require_schema_type(info.python_type, info)
                result[plan.alias] = _logical_mapping(nested_cls, columns)
            else:
                lengths = {
                    len(cast(list[Any], value)) for value in columns.values() if value is not None
                }
                if len(lengths) != 1:
                    raise TypeError(
                        f"Flat ListStruct columns for {plan.alias!r} have different lengths"
                    )
                nested_cls = _require_schema_type(info.python_type, info)
                result[plan.alias] = [
                    _logical_mapping(
                        nested_cls,
                        {
                            name: None if values is None else cast(list[Any], values)[index]
                            for name, values in columns.items()
                        },
                    )
                    for index in range(lengths.pop())
                ]
        elif plan.alias in data:
            consumed.add(plan.alias)
            value = data[plan.alias]
            if isinstance(info, Struct) and value is not None:
                nested = _require_schema_type(info.python_type, info)
                result[plan.alias] = _logical_mapping(nested, cast(Mapping[str, Any], value))
            elif isinstance(info, ListStruct) and value is not None:
                nested = _require_schema_type(info.python_type, info)
                result[plan.alias] = [
                    _logical_mapping(nested, item) for item in cast(list[Mapping[str, Any]], value)
                ]
            else:
                result[plan.alias] = value
    result.update({name: value for name, value in data.items() if name not in consumed})
    return result


def _resolved_schema(
    schema_cls: type[Schema],
    rows: list[Schema],
    *,
    strict: bool,
    extra_schema: Mapping[str, Any] | None,
) -> pl.Schema:
    explicit = extra_schema or {}
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
                        extra_schema=cast(Mapping[str, Any] | None, nested_explicit),
                    )
                )
            )
        else:
            if isinstance(nested_explicit, Mapping):
                raise TypeError(f"Declared field {plan.alias!r} cannot contain extra fields")
            result[plan.alias] = info.polars_dtype

    extras = schema_cls.__extras_field__
    if extras is None:
        unknown = set(explicit) - set(result)
        if unknown:
            name = sorted(unknown)[0]
            raise TypeError(f"{schema_cls.__name__} does not declare Extras for {name!r}")
        return pl.Schema(result)
    extra_names = [name for name in explicit if name not in result]
    for row in rows:
        values = getattr(row, extras.name)
        conflicts = set(result) & set(values)
        if conflicts:
            raise TypeError(
                f"Extras conflict with declared field(s): {', '.join(sorted(conflicts))}"
            )
        extra_names.extend(name for name in values if name not in extra_names)
    for name in extra_names:
        dtype = explicit.get(name)
        if dtype is None:
            dtype = pl.Series(
                name,
                [
                    _serialize_value(getattr(row, extras.name).get(name), by_alias=True)
                    for row in rows
                ],
                strict=strict,
            ).dtype
        if isinstance(dtype, Mapping):
            raise TypeError(f"Extra column {name!r} requires a Polars dtype")
        result[name] = dtype
    return pl.Schema(result)


def _struct_fields(dtype: Any) -> pl.Schema:
    if not isinstance(dtype, pl.Struct):
        raise TypeError(f"Expected Struct dtype, got {dtype!r}")
    return pl.Schema({field.name: field.dtype for field in dtype.fields})


def _list_inner(dtype: Any) -> Any:
    if not isinstance(dtype, pl.List):
        raise TypeError(f"Expected List dtype, got {dtype!r}")
    return dtype.inner

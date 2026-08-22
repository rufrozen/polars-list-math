"""Uniform field discovery for dataclasses and typed NamedTuple models."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Any, cast, get_type_hints


@dataclass(frozen=True, slots=True)
class RecordField:
    """Logical Python field metadata needed by the model compiler."""

    name: str
    annotation: Any
    has_default: bool


def is_named_tuple(cls: type[Any]) -> bool:
    """Return whether a class has the standard NamedTuple runtime shape."""
    if not issubclass(cls, tuple):
        return False
    names = getattr(cls, "_fields", None)
    return isinstance(names, tuple) and all(isinstance(name, str) for name in names)


def is_typed_named_tuple(cls: type[Any]) -> bool:
    """Return whether a class is an annotated NamedTuple record."""
    if not is_named_tuple(cls):
        return False
    names = cast(tuple[str, ...], cls._fields)
    annotations = getattr(cls, "__annotations__", None)
    return isinstance(annotations, dict) and set(names) <= annotations.keys()


def is_record(cls: type[Any]) -> bool:
    """Return whether a class can act as a typed Polars model record."""
    return is_dataclass(cls) or is_typed_named_tuple(cls)


def record_fields(cls: type[Any]) -> tuple[RecordField, ...]:
    """Return ordered annotated fields and their default availability."""
    hints = get_type_hints(cls, include_extras=True)
    if is_dataclass(cls):
        return tuple(
            RecordField(
                item.name,
                hints[item.name],
                item.default is not MISSING or item.default_factory is not MISSING,
            )
            for item in fields(cast(Any, cls))
        )
    if is_typed_named_tuple(cls):
        defaults = cast(dict[str, Any], getattr(cls, "_field_defaults", {}))
        return tuple(
            RecordField(name, hints[name], name in defaults)
            for name in cast(tuple[str, ...], cls._fields)
        )
    raise TypeError(f"{cls.__name__} must be a dataclass or typed NamedTuple")

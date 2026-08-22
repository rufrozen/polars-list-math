"""Root model decorator."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass
from typing import Callable, cast, dataclass_transform

from ._records import is_named_tuple, is_typed_named_tuple
from .builder import Builder
from .schema import Schema


@dataclass_transform()
def model[T: type](
    *,
    schema: type[Schema],
    strict: bool = False,
) -> Callable[[T], T]:
    """Validate a root record and cache its schema-driven builder."""
    if not isinstance(schema, type) or not issubclass(schema, Schema):
        raise TypeError("schema must be a typed Polars Schema class")
    if not isinstance(strict, bool):
        raise TypeError("strict must be a bool")
    return lambda cls: _decorate_model(cls, schema=schema, strict=strict)


def _decorate_model[T: type](
    cls: T,
    *,
    schema: type[Schema],
    strict: bool,
) -> T:
    if is_named_tuple(cls) and not is_typed_named_tuple(cls):
        raise TypeError(f"{cls.__name__} must be a typed NamedTuple")
    if not is_typed_named_tuple(cls) and (
        not is_dataclass(cls) or "__dataclass_fields__" not in cls.__dict__
    ):
        cls = cast(T, dataclass(slots=True)(cls))
    return cast(T, Builder(schema, cls, strict=strict).model)

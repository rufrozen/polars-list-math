"""Typed projections over :mod:`polars_list_math.typed_polars` columns."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Self, cast, overload

import polars as pl

from .model import Column

__all__ = ["View", "ViewField"]


class ViewField[T]:
    """A typed view attribute sourced from a schema column."""

    def __init__(self, source: Column[Any], *, alias: str | None = None) -> None:
        if not isinstance(source, Column):
            raise TypeError("ViewField source must be a Column")
        self.source = source
        self.alias_override = alias
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @property
    def alias(self) -> str:
        """Name of the projected DataFrame column."""
        return self.alias_override or self.name

    @property
    def dtype(self) -> Any:
        return self.source.dtype

    def expr(self) -> pl.Expr:
        return self.source.expr().alias(self.alias)

    @overload
    def __get__(self, instance: None, owner: type[View] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: View, owner: type[View] | None = None) -> T: ...

    def __get__(self, instance: View | None, owner: type[View] | None = None) -> T | Self:
        if instance is None:
            return self
        try:
            return cast(T, instance.__dict__[self.name])
        except KeyError:
            raise AttributeError(self.name) from None

    def __set__(self, instance: View, value: T) -> None:
        cast(dict[str, Any], instance.__dict__)[self.name] = value


class ViewMeta(type):
    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> ViewMeta:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        fields: dict[str, ViewField[Any]] = {}
        for base in bases:
            fields.update(getattr(base, "__view_fields__", {}))
        for field_name, value in namespace.items():
            if isinstance(value, ViewField):
                fields[field_name] = value

        aliases: dict[str, str] = {}
        for field_name, field in fields.items():
            previous = aliases.get(field.alias)
            if previous is not None and previous != field_name:
                raise TypeError(
                    f"{name}: duplicate view alias {field.alias!r} "
                    f"for {previous!r} and {field_name!r}"
                )
            aliases[field.alias] = field_name

        type.__setattr__(cls, "__view_fields__", fields)
        return cls


class View(metaclass=ViewMeta):
    """Base class for a typed Polars DataFrame projection."""

    __view_fields__: ClassVar[dict[str, ViewField[Any]]]

    def __init__(self, **kwargs: Any) -> None:
        fields = type(self).__view_fields__
        unknown = set(kwargs) - set(fields)
        missing = set(fields) - set(kwargs)
        if unknown:
            raise TypeError(
                f"Unexpected field(s) for {type(self).__name__}: {', '.join(sorted(unknown))}"
            )
        if missing:
            raise TypeError(
                f"Missing required field(s) for {type(self).__name__}: {', '.join(sorted(missing))}"
            )
        for name, value in kwargs.items():
            setattr(self, name, value)

    @classmethod
    def model_fields(cls) -> Mapping[str, ViewField[Any]]:
        return MappingProxyType(cls.__view_fields__)

    @classmethod
    def expressions(cls) -> tuple[pl.Expr, ...]:
        return tuple(field.expr() for field in cls.__view_fields__.values())

    @classmethod
    def select(cls, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Select view fields, replacing unavailable sources with nulls."""
        schema = frame.collect_schema()
        expressions = tuple(
            _safe_expression(field, schema) for field in cls.__view_fields__.values()
        )
        return frame.select(expressions)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> list[Self]:
        return list(cls.iter_frame(frame))

    @classmethod
    def iter_frame(cls, frame: pl.DataFrame) -> Iterator[Self]:
        selected = cls.select(frame)
        assert isinstance(selected, pl.DataFrame)
        aliases = {field.alias: name for name, field in cls.__view_fields__.items()}
        for row in selected.iter_rows(named=True):
            yield cls(**{aliases[key]: value for key, value in row.items()})

    def to_dict(self, *, by_alias: bool = False) -> dict[str, Any]:
        return {
            (field.alias if by_alias else name): getattr(self, name)
            for name, field in type(self).__view_fields__.items()
        }

    def __repr__(self) -> str:
        values = ", ".join(f"{name}={getattr(self, name)!r}" for name in type(self).__view_fields__)
        return f"{type(self).__name__}({values})"

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and isinstance(other, View)
            and self.to_dict() == other.to_dict()
        )


def _safe_expression(field: ViewField[Any], schema: pl.Schema) -> pl.Expr:
    if _source_exists(field.source, schema):
        return field.expr()

    dtype = field.source.dtype
    for _ in range(field.source._list_depth):
        dtype = pl.List(dtype)
    return pl.repeat(None, pl.len(), dtype=dtype).alias(field.alias)


def _source_exists(source: Column[Any], schema: pl.Schema) -> bool:
    try:
        dtype = schema[source.root_alias]
    except KeyError:
        return False

    for kind, alias in source._steps:
        if kind == "struct":
            if not isinstance(dtype, pl.Struct):
                return False
        elif kind == "list_item":
            if not isinstance(dtype, pl.List) or not isinstance(dtype.inner, pl.Struct):
                return False
            dtype = dtype.inner
        else:
            return False

        fields = {nested.name: nested.dtype for nested in dtype.fields}
        try:
            dtype = fields[alias]
        except KeyError:
            return False

    return True

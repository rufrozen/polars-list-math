"""Typed definitions for complete logical DataFrame schemas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Self,
    cast,
    get_args,
)

import polars as pl

from ._binding import get_builder
from .dtypes import annotation_to_dtype


class Column[T]:
    """A typed scalar or list column in a logical DataFrame schema."""

    def __init__(self, *, polars_name: str | None = None, dtype: Any | None = None) -> None:
        self.name = ""
        self.polars_name = polars_name or ""
        self.dtype = dtype
        self._python_type: Any | None = None
        self._root = ""
        self._steps: tuple[tuple[str, str], ...] = ()

    @property
    def python_type(self) -> Any:
        if self._python_type is None:
            args = get_args(getattr(self, "__orig_class__", None))
            if len(args) != 1:
                raise TypeError(f"{type(self).__name__} {self.name!r} requires a generic type")
            self._python_type = args[0]
        return self._python_type

    def expr(self) -> pl.Expr:
        """Build a Polars expression for this physical field path."""
        return _apply_steps(pl.col(self._root), self._steps)

    def _bind(
        self,
        *,
        name: str,
        root: str,
        steps: tuple[tuple[str, str], ...],
    ) -> Self:
        self.name = name
        if not self.polars_name:
            self.polars_name = name
        if self.dtype is None:
            self.dtype = annotation_to_dtype(self.python_type)
        self._root = root
        self._steps = steps
        return self

    def _bound_copy(
        self,
        *,
        root: str,
        steps: tuple[tuple[str, str], ...],
    ) -> Column[Any]:
        result = Column[Any](polars_name=self.polars_name, dtype=self.dtype)
        result.name = self.name
        result._python_type = self.python_type
        result._root = root
        result._steps = steps
        return result

    def __repr__(self) -> str:
        """Return concise column metadata for debugging."""
        return f"Column(polars_name={self.polars_name!r}, dtype={self.dtype!r})"


class Struct[S: Schema](Column[S]):
    """A typed Polars Struct column with a nested schema."""

    def __init__(
        self,
        *,
        polars_name: str | None = None,
        flat: bool = False,
        flat_divider: str = "_",
    ) -> None:
        super().__init__(polars_name=polars_name)
        _validate_flat_options(flat, flat_divider)
        self.flat = flat
        self.flat_divider = flat_divider
        self._fields: _BoundSchema | None = None

    @property
    def schema(self) -> type[S]:
        schema = self.python_type
        if not isinstance(schema, type) or not issubclass(schema, Schema):
            raise TypeError(f"Struct {self.name!r} requires a Schema type")
        return cast(type[S], schema)

    @property
    def fields(self) -> type[S]:
        if self._fields is None:
            raise RuntimeError(f"Struct {self.name!r} is not bound")
        return cast(type[S], self._fields)

    def _bind(
        self,
        *,
        name: str,
        root: str,
        steps: tuple[tuple[str, str], ...],
    ) -> Self:
        self.name = name
        if not self.polars_name:
            self.polars_name = name
        self.dtype = pl.Struct(self.schema.polars_schema())
        self._root = root
        self._steps = steps
        self._fields = (
            _bind_flat_schema(self.schema, self.polars_name, self.flat_divider, None)
            if self.flat and not steps
            else _bind_schema(self.schema, root, steps, "struct")
        )
        return self

    def _bound_copy(
        self,
        *,
        root: str,
        steps: tuple[tuple[str, str], ...],
    ) -> Struct[Any]:
        result = Struct[Any](
            polars_name=self.polars_name, flat=self.flat, flat_divider=self.flat_divider
        )
        result.name = self.name
        result._python_type = self.schema
        result.dtype = self.dtype
        result._root = root
        result._steps = steps
        result._fields = _bind_schema(self.schema, root, steps, "struct")
        return result


class ListStruct[S: Schema](Column[list[S]]):
    """A typed Polars List[Struct] column with an item schema."""

    def __init__(
        self,
        *,
        polars_name: str | None = None,
        flat: bool = False,
        flat_divider: str = "_",
    ) -> None:
        super().__init__(polars_name=polars_name)
        _validate_flat_options(flat, flat_divider)
        self.flat = flat
        self.flat_divider = flat_divider
        self._item: _BoundSchema | None = None

    @property
    def schema(self) -> type[S]:
        args = get_args(self.python_type)
        schema = args[0] if len(args) == 1 else None
        if not isinstance(schema, type) or not issubclass(schema, Schema):
            raise TypeError(f"ListStruct {self.name!r} requires a Schema type")
        return cast(type[S], schema)

    @property
    def item(self) -> type[S]:
        if self._item is None:
            raise RuntimeError(f"ListStruct {self.name!r} is not bound")
        return cast(type[S], self._item)

    @property
    def python_type(self) -> Any:
        if self._python_type is None:
            args = get_args(getattr(self, "__orig_class__", None))
            if len(args) != 1:
                raise TypeError(f"ListStruct {self.name!r} requires a generic type")
            self._python_type = list[args[0]]
        return self._python_type

    def _bind(
        self,
        *,
        name: str,
        root: str,
        steps: tuple[tuple[str, str], ...],
    ) -> Self:
        self.name = name
        if not self.polars_name:
            self.polars_name = name
        self.dtype = pl.List(pl.Struct(self.schema.polars_schema()))
        self._root = root
        self._steps = steps
        self._item = (
            _bind_flat_schema(self.schema, self.polars_name, self.flat_divider, "list")
            if self.flat and not steps
            else _bind_schema(self.schema, root, steps, "list_struct")
        )
        return self

    def _bound_copy(
        self,
        *,
        root: str,
        steps: tuple[tuple[str, str], ...],
    ) -> ListStruct[Any]:
        result = ListStruct[Any](
            polars_name=self.polars_name, flat=self.flat, flat_divider=self.flat_divider
        )
        result.name = self.name
        result._python_type = list[self.schema]
        result.dtype = self.dtype
        result._root = root
        result._steps = steps
        result._item = _bind_schema(self.schema, root, steps, "list_struct")
        return result


class SchemaMeta(type):
    """Collect and bind typed column declarations."""

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        schema: dict[str, Any],
        **kwargs: Any,
    ) -> SchemaMeta:
        cls = super().__new__(mcls, name, bases, schema, **kwargs)
        columns: dict[str, Column[Any]] = {}
        for base in bases:
            columns.update(getattr(base, "__schema_columns__", {}))
        columns.update(
            (field_name, value) for field_name, value in schema.items() if isinstance(value, Column)
        )

        polars_names: set[str] = set()
        for field_name, column in columns.items():
            polars_name = column.polars_name or field_name
            if polars_name in polars_names:
                raise TypeError(f"{name} has duplicate Polars name {polars_name!r}")
            polars_names.add(polars_name)
            column._bind(name=field_name, root=polars_name, steps=())

        type.__setattr__(cls, "__schema_columns__", columns)
        return cls


class Schema(metaclass=SchemaMeta):
    """Base class for a complete logical DataFrame column schema."""

    __schema_columns__: ClassVar[dict[str, Column[Any]]]

    def __new__(cls) -> Self:
        raise TypeError(f"{cls.__name__} is a schema and cannot be instantiated")

    @classmethod
    def fields(cls) -> Mapping[str, Column[Any]]:
        """Return immutable Python-name to column metadata mapping."""
        return MappingProxyType(cls.__schema_columns__)

    @classmethod
    def polars_schema(cls) -> pl.Schema:
        """Return the complete physical Polars schema."""
        return pl.Schema(_physical_fields(cls))

    @classmethod
    def model_schema(cls, model: type[Any]) -> pl.Schema:
        """Return the cached physical schema selected by a root model."""
        return get_builder(model, schema=cls).polars_schema()

    @classmethod
    def to_frame(cls, row: Any, *, strict: bool = True) -> pl.DataFrame:
        """Serialize one validated root model value."""
        return get_builder(type(row), schema=cls).to_frame(row, strict=strict)

    @classmethod
    def to_frame_many[T](
        cls,
        model: type[T],
        rows: Iterable[T],
        *,
        strict: bool = True,
    ) -> pl.DataFrame:
        """Serialize validated root model values."""
        return get_builder(model, schema=cls).to_frame_many(rows, strict=strict)

    @classmethod
    def iter_frame[T](
        cls,
        model: type[T],
        frame: pl.DataFrame,
        *,
        strict_schema: bool = False,
    ) -> Iterable[T]:
        """Deserialize DataFrame rows into the explicit root model."""
        return get_builder(model, schema=cls).iter_frame(model, frame, strict_schema=strict_schema)

    @classmethod
    def from_frame[T](
        cls,
        model: type[T],
        frame: pl.DataFrame,
        *,
        strict_schema: bool = False,
    ) -> T:
        """Deserialize a one-row DataFrame into the explicit root model."""
        return get_builder(model, schema=cls).from_frame(model, frame, strict_schema=strict_schema)

    @classmethod
    def assert_frame_schema(cls, model: type[Any], frame: pl.DataFrame) -> None:
        """Require the exact physical schema selected by the explicit model."""
        get_builder(model, schema=cls).assert_frame_schema(frame)

    @classmethod
    def to_dict(cls, row: Any, *, by_polars_name: bool = False) -> dict[str, Any]:
        """Serialize one root model to a dictionary."""
        return get_builder(type(row), schema=cls).to_dict(row, by_polars_name=by_polars_name)

    @classmethod
    def from_dict[T](
        cls,
        model: type[T],
        data: Mapping[str, Any],
        *,
        by_polars_name: bool = False,
    ) -> T:
        """Deserialize a dictionary into the explicit root model type."""
        return get_builder(model, schema=cls).from_dict(model, data, by_polars_name=by_polars_name)


class _BoundSchema:
    def __init__(self, fields: Mapping[str, Column[Any]]) -> None:
        self._fields = dict(fields)

    def __getattr__(self, name: str) -> Column[Any]:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name: str) -> Column[Any]:
        return self._fields[name]


def _bind_schema(
    schema: type[Schema],
    root: str,
    steps: tuple[tuple[str, str], ...],
    container_kind: str,
) -> _BoundSchema:
    fields: dict[str, Column[Any]] = {}
    for name, column in schema.__schema_columns__.items():
        nested_steps = steps + ((container_kind, column.polars_name),)
        fields[name] = column._bound_copy(root=root, steps=nested_steps)
    return _BoundSchema(fields)


def _bind_flat_schema(
    schema: type[Schema], prefix: str, divider: str, list_kind: str | None
) -> _BoundSchema:
    fields: dict[str, Column[Any]] = {}
    for name, column in schema.__schema_columns__.items():
        root = f"{prefix}{divider}{column.polars_name}"
        steps: tuple[tuple[str, str], ...] = ()
        if list_kind is not None:
            # A flattened ListStruct child is already a physical list column.
            steps = ()
        fields[name] = column._bound_copy(root=root, steps=steps)
    return _BoundSchema(fields)


def _physical_fields(schema: type[Schema]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in schema.__schema_columns__.values():
        if isinstance(column, (Struct, ListStruct)) and column.flat:
            for child_name, child_dtype in _physical_fields(column.schema).items():
                name = f"{column.polars_name}{column.flat_divider}{child_name}"
                dtype = pl.List(child_dtype) if isinstance(column, ListStruct) else child_dtype
                _add_physical_field(result, name, dtype, schema)
        else:
            _add_physical_field(result, column.polars_name, cast(Any, column.dtype), schema)
    return result


def _add_physical_field(
    fields: dict[str, Any], name: str, dtype: Any, schema: type[Schema]
) -> None:
    if name in fields:
        raise TypeError(f"{schema.__name__} has conflicting physical field {name!r}")
    fields[name] = dtype


def _validate_flat_options(flat: bool, flat_divider: str) -> None:
    if not isinstance(flat, bool):
        raise TypeError("flat must be a bool")
    if not isinstance(flat_divider, str) or not flat_divider:
        raise TypeError("flat_divider must be a non-empty string")


def _apply_steps(expr: pl.Expr, steps: tuple[tuple[str, str], ...]) -> pl.Expr:
    if not steps:
        return expr
    (kind, name), remaining = steps[0], steps[1:]
    if kind == "struct":
        return _apply_steps(expr.struct.field(name), remaining)
    if kind == "list_struct":
        nested = _apply_steps(pl.element().struct.field(name), remaining)
        return expr.list.eval(nested)
    raise RuntimeError(f"Unknown schema path step: {kind}")

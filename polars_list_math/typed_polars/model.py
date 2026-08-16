"""Typed row models and schemas for Polars.

Python 3.12+.

The same declaration provides:
- typed instance attributes through descriptors;
- lintable class-level Polars column references;
- aliases for physical Polars names;
- exact ``pl.Schema`` generation;
- nested Struct/List[Struct] expressions;
- row <-> dict <-> DataFrame conversion.

Example::

    class Suggestion(Schema):
        value = Field[str]()
        search_corrected_query = Field[str](alias="searchCorrectedQuery")

    class Completion(Schema):
        prefix = Field[str](alias="queryPrefix")
        suggestions = ListStruct[Suggestion]()

    class Row(Schema):
        request_id = Field[str](alias="requestId")
        position = Field[I32]()
        completion = Struct[Completion](alias="completionData")

    row.request_id                         # str
    Row.request_id                        # Column[str]
    Row.completion                        # StructColumn[Completion]
    Row.completion.fields.prefix          # Column[str]
    Row.completion.fields.suggestions     # ListStructColumn[Suggestion]
    Row.completion.fields.suggestions.item.value  # Column[str]

A deliberate typing trade-off: because Python's static type system has no
mapped types, a runtime base class cannot synthesize a statically exact
``__init__`` signature from descriptor declarations. Field access and column
paths are statically typed, while constructor keyword arguments are checked at
runtime. Full constructor-keyword checking additionally requires codegen or a
type-checker plugin.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Self,
    cast,
    get_args,
    get_origin,
    overload,
)

import polars as pl

from .dtypes import (
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    DurationMs,
    DurationNs,
    DurationUs,
    TimestampMs,
    TimestampNs,
    TimestampUs,
    annotation_to_dtype,
    base_annotation,
)

__all__ = [
    "F32",
    "F64",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
    "Column",
    "DurationMs",
    "DurationNs",
    "DurationUs",
    "Extras",
    "Field",
    "Include",
    "IncludeListStruct",
    "IncludeStruct",
    "ListStruct",
    "ListStructColumn",
    "Schema",
    "Struct",
    "StructColumn",
    "TimestampMs",
    "TimestampNs",
    "TimestampUs",
]


_MISSING = object()


# ---------------------------------------------------------------------------
# Polars column references
# ---------------------------------------------------------------------------


class Column[T]:
    """A typed reference to a top-level or nested Polars field."""

    def __init__(
        self,
        *,
        name: str,
        alias: str,
        dtype: Any,
        root_alias: str,
        steps: tuple[tuple[str, str], ...] = (),
        python_path: tuple[str, ...] = (),
        polars_path: tuple[str, ...] = (),
        flat_path: tuple[str, ...] = (),
        field_info: _FieldBase | None = None,
    ) -> None:
        self.name = name
        self.alias = alias
        self.dtype = dtype
        self.root_alias = root_alias
        self._steps = steps
        self.python_path = python_path or (name,)
        self.polars_path = polars_path or (alias,)
        self.flat_path = flat_path or (alias,)
        self._field_info = field_info

    @property
    def flat_name(self) -> str:
        return ":".join(self.flat_path)

    def nested_expr(self) -> pl.Expr:
        return _build_expr(self.root_alias, self._steps)

    def flat_expr(self) -> pl.Expr:
        return pl.col(self.flat_name)

    def __str__(self) -> str:
        return self.alias

    def __repr__(self) -> str:
        return f"Column(path={'.'.join(self.polars_path)!r}, dtype={self.dtype!r})"


class _BoundColumnsProxy:
    """Runtime proxy for nested fields.

    Public properties expose it as ``type[SchemaSubclass]`` to static type
    checkers. At runtime it returns columns already bound to the parent path.
    """

    def __init__(self, columns: Mapping[str, Column[Any]]) -> None:
        self._columns = dict(columns)

    def __getattr__(self, name: str) -> Column[Any]:
        try:
            return self._columns[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name: str) -> Column[Any]:
        return self._columns[name]

    def __iter__(self) -> Iterator[Column[Any]]:
        return iter(self._columns.values())

    def __repr__(self) -> str:
        return f"BoundColumns({', '.join(self._columns)})"


class StructColumn[S: Schema](Column[S]):
    """A Struct column. Use ``.fields.<name>`` for typed nested access."""

    def __init__(self, *, fields_proxy: _BoundColumnsProxy, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._fields_proxy = fields_proxy

    @property
    def fields(self) -> type[S]:
        # Static trick: the Schema class contains the same typed descriptors,
        # while runtime needs a path-bound proxy rather than the real class.
        return cast(type[S], self._fields_proxy)

    def flat_expr(self) -> pl.Expr:
        raise TypeError("Struct has no single column in flat representation")


class ListStructColumn[S: Schema](Column[list[S]]):
    """A List[Struct] column. Use ``.item.<name>`` for typed item access."""

    def __init__(self, *, item_proxy: _BoundColumnsProxy, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._item_proxy = item_proxy

    @property
    def item(self) -> type[S]:
        return cast(type[S], self._item_proxy)

    def flat_expr(self) -> pl.Expr:
        raise TypeError("ListStruct has no single column in flat representation")


# ---------------------------------------------------------------------------
# Typed field descriptors
# ---------------------------------------------------------------------------


class _FieldBase:
    def __init__(
        self,
        default: Any = _MISSING,
        *,
        alias: str | None = None,
        flat_alias: str | None = None,
        dtype: Any | None = None,
        default_factory: Callable[[], Any] | object = _MISSING,
        repr: bool = True,  # noqa: A002 - matches dataclasses/Pydantic terminology
    ) -> None:
        if default is not _MISSING and default_factory is not _MISSING:
            raise TypeError("field cannot specify both default and default_factory")

        self.alias_override = alias
        self.flat_alias = flat_alias
        self.dtype_override = dtype
        self.default = default
        self.default_factory = default_factory
        self.repr = repr

        self.name = ""
        self.index = -1
        self.owner: type[Schema] | None = None
        self._column: Column[Any] | None = None
        self._explicit_type: Any = _MISSING
        self._resolved_python_type: Any = _MISSING

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @property
    def alias(self) -> str:
        return self.alias_override or self.name

    @property
    def required(self) -> bool:
        return self.default is _MISSING and self.default_factory is _MISSING

    @property
    def python_type(self) -> Any:
        if self._explicit_type is not _MISSING:
            return self._explicit_type
        if self._resolved_python_type is not _MISSING:
            return self._resolved_python_type

        orig = getattr(self, "__orig_class__", None)
        if orig is None:
            raise TypeError(
                f"{type(self).__name__} {self.name!r} must have a generic type, "
                f"for example {type(self).__name__}[str]()"
            )

        args = get_args(orig)
        if len(args) != 1:
            raise TypeError(f"Cannot resolve generic type for {self!r}")
        self._resolved_python_type = args[0]
        return self._resolved_python_type

    @property
    def polars_dtype(self) -> Any:
        if self.dtype_override is not None:
            return self.dtype_override
        return annotation_to_dtype(self.storage_type)

    @property
    def storage_type(self) -> Any:
        return self.python_type

    def get_default(self) -> Any:
        if self.default is not _MISSING:
            return copy.deepcopy(self.default)
        if self.default_factory is not _MISSING:
            factory = self.default_factory
            assert callable(factory)
            return factory()
        raise TypeError(f"Field {self.name!r} has no default")

    def _bind(self, owner: type[Schema]) -> None:
        self.owner = owner

    def _set_column(self, column: Column[Any]) -> None:
        self._column = column

    def _column_or_error(self) -> Column[Any]:
        if self._column is None:
            raise RuntimeError(f"Field {self.name!r} is not bound to a Schema")
        return self._column

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, alias={self.alias!r}, "
            f"dtype={self.polars_dtype!r}, required={self.required!r})"
        )


class Field[T](_FieldBase):
    """Scalar/list field descriptor.

    ``obj.x`` is statically ``T``; ``Model.x`` is statically ``Column[T]``.
    """

    @overload
    def __get__(self, instance: None, owner: type[Schema] | None = None) -> Column[T]: ...

    @overload
    def __get__(self, instance: Schema, owner: type[Schema] | None = None) -> T: ...

    def __get__(self, instance: Schema | None, owner: type[Schema] | None = None) -> T | Column[T]:
        if instance is None:
            return cast(Column[T], self._column_or_error())
        try:
            value = instance._values[self.index]
        except IndexError:
            raise AttributeError(self.name) from None
        if value is _MISSING:
            raise AttributeError(self.name)
        return cast(T, value)

    def __set__(self, instance: Schema, value: T) -> None:
        instance._values[self.index] = value


class Struct[S: Schema](_FieldBase):
    """Nested Schema stored as a Polars Struct."""

    @overload
    def __get__(self, instance: None, owner: type[Schema] | None = None) -> StructColumn[S]: ...

    @overload
    def __get__(self, instance: Schema, owner: type[Schema] | None = None) -> S: ...

    def __get__(
        self, instance: Schema | None, owner: type[Schema] | None = None
    ) -> S | StructColumn[S]:
        if instance is None:
            return cast(StructColumn[S], self._column_or_error())
        try:
            value = instance._values[self.index]
        except IndexError:
            raise AttributeError(self.name) from None
        if value is _MISSING:
            raise AttributeError(self.name)
        return cast(S, value)

    def __set__(self, instance: Schema, value: S) -> None:
        instance._values[self.index] = value

    @property
    def storage_type(self) -> Any:
        return self.python_type


class ListStruct[S: Schema](_FieldBase):
    """List of nested Schema objects stored as Polars List[Struct]."""

    @overload
    def __get__(self, instance: None, owner: type[Schema] | None = None) -> ListStructColumn[S]: ...

    @overload
    def __get__(self, instance: Schema, owner: type[Schema] | None = None) -> list[S]: ...

    def __get__(
        self, instance: Schema | None, owner: type[Schema] | None = None
    ) -> list[S] | ListStructColumn[S]:
        if instance is None:
            return cast(ListStructColumn[S], self._column_or_error())
        try:
            value = instance._values[self.index]
        except IndexError:
            raise AttributeError(self.name) from None
        if value is _MISSING:
            raise AttributeError(self.name)
        return cast(list[S], value)

    def __set__(self, instance: Schema, value: list[S]) -> None:
        instance._values[self.index] = value

    @property
    def storage_type(self) -> Any:
        return list[self.python_type]


class Include[T](Field[T]):
    """A scalar/list field projected from another Schema column."""

    def __init__(self, source: Column[T]) -> None:
        if isinstance(source, (StructColumn, ListStructColumn)):
            raise TypeError("Use IncludeStruct or IncludeListStruct for nested schemas")
        info = _source_field_info(source)
        super().__init__(
            _copied_default(info),
            alias=source.alias,
            dtype=source.dtype,
            default_factory=_copied_default_factory(info),
            repr=info.repr,
        )
        self._explicit_type = info.python_type


class IncludeStruct[P: Schema](Struct[P]):
    """A Struct field projected into an explicitly declared nested Schema."""

    def __init__(self, source: StructColumn[Any], schema: type[P]) -> None:
        if not isinstance(source, StructColumn):
            raise TypeError("IncludeStruct source must be a StructColumn")
        info = _source_field_info(source)
        super().__init__(
            _copied_default(info),
            alias=source.alias,
            flat_alias=info.flat_alias,
            default_factory=_copied_default_factory(info),
            repr=info.repr,
        )
        self._explicit_type = schema


class IncludeListStruct[P: Schema](ListStruct[P]):
    """A ListStruct field projected into an explicitly declared item Schema."""

    def __init__(self, source: ListStructColumn[Any], schema: type[P]) -> None:
        if not isinstance(source, ListStructColumn):
            raise TypeError("IncludeListStruct source must be a ListStructColumn")
        info = _source_field_info(source)
        super().__init__(
            _copied_default(info),
            alias=source.alias,
            flat_alias=info.flat_alias,
            default_factory=_copied_default_factory(info),
            repr=info.repr,
        )
        self._explicit_type = schema


def _source_field_info(source: Column[Any]) -> _FieldBase:
    if not isinstance(source, Column):
        raise TypeError("Include source must be a Column")
    if source._field_info is None:
        raise TypeError("Include source must belong to a Schema")
    return source._field_info


def _copied_default(info: _FieldBase) -> Any:
    if info.default is _MISSING:
        return _MISSING
    return copy.deepcopy(info.default)


def _copied_default_factory(info: _FieldBase) -> Callable[[], Any] | object:
    return info.default_factory


@dataclass(frozen=True, slots=True)
class _FieldPlan:
    index: int
    name: str
    alias: str
    info: _FieldBase
    serialize_python: Callable[[Any], Any]
    serialize_alias: Callable[[Any], Any]


def _compile_field_plan(index: int, name: str, info: _FieldBase) -> _FieldPlan:
    return _FieldPlan(
        index=index,
        name=name,
        alias=info.alias,
        info=info,
        serialize_python=_compile_serializer(info.storage_type, by_alias=False),
        serialize_alias=_compile_serializer(info.storage_type, by_alias=True),
    )


def _compile_serializer(
    annotation: Any,
    *,
    by_alias: bool,
) -> Callable[[Any], Any]:
    annotation = base_annotation(annotation)
    origin = get_origin(annotation)

    if isinstance(annotation, type) and issubclass(annotation, Schema):

        def serialize_schema(value: Any) -> Any:
            if value is None:
                return None
            if not isinstance(value, annotation):
                raise TypeError(f"Expected {annotation.__name__}, got {type(value).__name__}")
            return value._to_dict_with_plan(by_alias=by_alias)

        return serialize_schema

    if origin is list:
        (item_type,) = get_args(annotation)
        item_serializer = _compile_serializer(item_type, by_alias=by_alias)

        def serialize_list(value: Any) -> Any:
            if value is None:
                return None
            if not isinstance(value, list):
                raise TypeError(f"Expected list, got {type(value).__name__}")
            return [item_serializer(item) for item in value]

        return serialize_list

    if origin is dict:
        key_type, value_type = get_args(annotation)
        key_serializer = _compile_serializer(key_type, by_alias=by_alias)
        value_serializer = _compile_serializer(value_type, by_alias=by_alias)

        def serialize_dict(value: Any) -> Any:
            if value is None:
                return None
            if not isinstance(value, Mapping):
                raise TypeError(f"Expected mapping, got {type(value).__name__}")
            return [
                {
                    "key": key_serializer(key),
                    "value": value_serializer(item),
                }
                for key, item in value.items()
            ]

        return serialize_dict

    def serialize_scalar(value: Any) -> Any:
        return value

    return serialize_scalar


class Extras:
    """Virtual descriptor that captures fields not declared by a Schema."""

    def __init__(self) -> None:
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type[Schema] | None = None) -> Self: ...

    @overload
    def __get__(self, instance: Schema, owner: type[Schema] | None = None) -> dict[str, Any]: ...

    def __get__(
        self, instance: Schema | None, owner: type[Schema] | None = None
    ) -> Self | dict[str, Any]:
        if instance is None:
            return self
        return cast(dict[str, Any], instance.__dict__[self.name])

    def __set__(self, instance: Schema, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise TypeError(f"Extras {self.name!r} must be a mapping")
        cast(dict[str, Any], instance.__dict__)[self.name] = dict(value)


# ---------------------------------------------------------------------------
# Schema metaclass and row model
# ---------------------------------------------------------------------------


class SchemaMeta(type):
    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> SchemaMeta:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        field_bases = [base for base in bases if getattr(base, "__schema_fields__", None)]
        if len(field_bases) > 1:
            base_names = ", ".join(base.__name__ for base in field_bases)
            raise TypeError(
                f"{name}: multiple inheritance from field-bearing schemas "
                f"is not supported ({base_names})"
            )

        fields: dict[str, _FieldBase] = {}
        extras_field: Extras | None = None
        for base in bases:
            fields.update(getattr(base, "__schema_fields__", {}))
            inherited_extras = getattr(base, "__extras_field__", None)
            if inherited_extras is not None:
                extras_field = inherited_extras

        for attr_name, value in namespace.items():
            if isinstance(value, _FieldBase):
                if value.flat_alias is not None and not isinstance(value, (Struct, ListStruct)):
                    raise TypeError(
                        f"{name}.{attr_name}: flat_alias is only supported by Struct and ListStruct"
                    )
                value._bind(cast("type[Schema]", cls))
                fields[attr_name] = value
            elif isinstance(value, Extras):
                if extras_field is not None and extras_field is not value:
                    raise TypeError(f"{name}: only one Extras field is allowed")
                extras_field = value

        aliases: dict[str, str] = {}
        for field_name, info in fields.items():
            other = aliases.get(info.alias)
            if other is not None and other != field_name:
                raise TypeError(
                    f"{name}: duplicate Polars alias {info.alias!r} "
                    f"for {other!r} and {field_name!r}"
                )
            aliases[info.alias] = field_name

        field_plan = tuple(
            _compile_field_plan(index, field_name, info)
            for index, (field_name, info) in enumerate(fields.items())
        )
        type.__setattr__(cls, "__schema_fields__", fields)
        type.__setattr__(cls, "__extras_field__", extras_field)
        type.__setattr__(cls, "__polars_schema_cache__", None)
        type.__setattr__(cls, "__field_plan__", field_plan)
        type.__setattr__(
            cls,
            "__field_plan_by_alias__",
            {plan.alias: plan for plan in field_plan},
        )

        # Bind class-level typed Column objects. This happens only after the
        # complete Schema subclass exists, so nested schema descriptors can be
        # inspected safely.
        for index, info in enumerate(fields.values()):
            info.index = index
            info._set_column(
                _build_column(
                    info,
                    root_alias=info.alias,
                    steps=(),
                    python_path=(info.name,),
                    polars_path=(info.alias,),
                    flat_path=(info.flat_alias or info.alias,),
                )
            )

        return cls

    @property
    def schema(cls) -> pl.Schema:
        return cast("type[Schema]", cls).polars_schema()


class Schema(metaclass=SchemaMeta):
    """Base row model.

    Constructor keys are validated at runtime. Descriptor reads/writes and
    class-level column references are statically typed.
    """

    __schema_fields__: ClassVar[dict[str, _FieldBase]]
    __extras_field__: ClassVar[Extras | None]
    __polars_schema_cache__: ClassVar[pl.Schema | None]
    __field_plan__: ClassVar[tuple[_FieldPlan, ...]]
    __field_plan_by_alias__: ClassVar[dict[str, _FieldPlan]]

    _values: list[Any]

    def __init__(self, **kwargs: Any) -> None:
        fields = type(self).__schema_fields__
        self._values = [_MISSING] * len(fields)
        extras_field = type(self).__extras_field__
        accepted = set(fields)
        if extras_field is not None:
            accepted.add(extras_field.name)
        unknown = set(kwargs) - accepted
        if unknown:
            raise TypeError(
                f"Unexpected field(s) for {type(self).__name__}: {', '.join(sorted(unknown))}"
            )

        missing: list[str] = []
        for name, info in fields.items():
            if name in kwargs:
                value = kwargs[name]
            elif info.required:
                missing.append(name)
                continue
            else:
                value = info.get_default()
            setattr(self, name, value)

        if extras_field is not None:
            setattr(self, extras_field.name, kwargs.get(extras_field.name, {}))

        if missing:
            raise TypeError(
                f"Missing required field(s) for {type(self).__name__}: {', '.join(missing)}"
            )

    @classmethod
    def model_fields(cls) -> Mapping[str, _FieldBase]:
        return MappingProxyType(cls.__schema_fields__)

    @classmethod
    def polars_schema(cls, extra_schema: Mapping[str, Any] | None = None) -> pl.Schema:
        if cls.__polars_schema_cache__ is None:
            cls.__polars_schema_cache__ = pl.Schema(
                {info.alias: info.polars_dtype for info in cls.__schema_fields__.values()}
            )
        if not extra_schema:
            return cls.__polars_schema_cache__
        return _apply_extra_schema(cls, extra_schema)

    def to_dict(self, *, by_alias: bool = False) -> dict[str, Any]:
        return self._to_dict_with_plan(by_alias=by_alias)

    def _to_dict_with_plan(self, *, by_alias: bool) -> dict[str, Any]:
        result = {
            (plan.alias if by_alias else plan.name): (
                plan.serialize_alias if by_alias else plan.serialize_python
            )(self._values[plan.index])
            for plan in type(self).__field_plan__
        }
        extras_field = type(self).__extras_field__
        if extras_field is not None:
            extras = getattr(self, extras_field.name)
            conflicts = set(result) & set(extras)
            if conflicts:
                raise TypeError(
                    f"Extras conflict with declared field(s): {', '.join(sorted(conflicts))}"
                )
            result.update(
                {key: _serialize_value(value, by_alias=by_alias) for key, value in extras.items()}
            )
        return result

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        by_alias: bool = False,
        forbid_extra: bool = True,
    ) -> Self:
        kwargs: dict[str, Any] = {}
        expected_keys: set[str] = set()
        missing: list[str] = []

        for name, info in cls.__schema_fields__.items():
            key = info.alias if by_alias else name
            expected_keys.add(key)

            if key in data:
                kwargs[name] = _deserialize_value(
                    info.storage_type,
                    data[key],
                    by_alias=by_alias,
                )
            elif info.required:
                missing.append(key)

        if missing:
            raise TypeError(f"Missing required key(s) for {cls.__name__}: {', '.join(missing)}")

        extra = set(data) - expected_keys
        if cls.__extras_field__ is not None:
            kwargs[cls.__extras_field__.name] = {key: data[key] for key in extra}
        elif forbid_extra and extra:
            raise TypeError(f"Unexpected key(s) for {cls.__name__}: {', '.join(sorted(extra))}")

        return cls(**kwargs)

    def to_frame(
        self,
        *,
        strict: bool = True,
        extra_schema: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        return type(self).to_frame_many(
            [self],
            strict=strict,
            extra_schema=extra_schema,
        )

    def to_flat_frame(
        self,
        *,
        strict: bool = True,
        extra_schema: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Serialize this row without Struct columns, using ``:`` paths."""
        return type(self).to_flat_frame_many(
            [self],
            strict=strict,
            extra_schema=extra_schema,
        )

    @classmethod
    def to_frame_many(
        cls,
        rows: Iterable[Self],
        *,
        strict: bool = True,
        extra_schema: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        from .frame import build_frame

        return build_frame(
            cls,
            list(rows),
            strict=strict,
            extra_schema=extra_schema,
        )

    @classmethod
    def to_flat_frame_many(
        cls,
        rows: Iterable[Self],
        *,
        strict: bool = True,
        extra_schema: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Serialize rows to scalar/list columns with colon-separated paths."""
        from .frame import build_flat_frame

        return build_flat_frame(
            cls,
            list(rows),
            strict=strict,
            extra_schema=extra_schema,
        )

    @classmethod
    def flat_schema(
        cls,
        extra_schema: Mapping[str, Any] | None = None,
    ) -> pl.Schema:
        """Return the flat schema when all dynamic fields are explicit."""
        from .frame import flatten_schema

        return flatten_schema(cls, cls.polars_schema(extra_schema))

    @classmethod
    def from_frame(
        cls,
        df: pl.DataFrame,
        index: int = 0,
        *,
        strict_schema: bool = False,
    ) -> Self:
        if strict_schema:
            cls.assert_frame_schema(df)
        row = df.row(index, named=True)
        return cls.from_dict(row, by_alias=True, forbid_extra=False)

    @classmethod
    def iter_frame(
        cls,
        df: pl.DataFrame,
        *,
        strict_schema: bool = False,
    ) -> Iterator[Self]:
        if strict_schema:
            cls.assert_frame_schema(df)
        for row in df.iter_rows(named=True):
            yield cls.from_dict(row, by_alias=True, forbid_extra=False)

    @classmethod
    def assert_frame_schema(cls, df: pl.DataFrame, *, allow_extra: bool = False) -> None:
        expected = cls.polars_schema()
        actual = df.schema

        if not allow_extra:
            if actual != expected:
                raise TypeError(
                    f"Unexpected DataFrame schema for {cls.__name__}:\n"
                    f"expected={expected}\nactual={actual}"
                )
            return

        missing = [name for name in expected if name not in actual]
        wrong = [
            name for name, dtype in expected.items() if name in actual and actual[name] != dtype
        ]
        if missing or wrong:
            raise TypeError(
                f"Unexpected DataFrame schema for {cls.__name__}: "
                f"missing={missing}, wrong_dtype={wrong}"
            )

    def __repr__(self) -> str:
        values = []
        for name, info in type(self).__schema_fields__.items():
            if info.repr:
                values.append(f"{name}={getattr(self, name)!r}")
        extras_field = type(self).__extras_field__
        if extras_field is not None:
            values.append(f"{extras_field.name}={getattr(self, extras_field.name)!r}")
        return f"{type(self).__name__}({', '.join(values)})"

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is type(other)
            and isinstance(other, Schema)
            and self.to_dict() == other.to_dict()
        )


# ---------------------------------------------------------------------------
# Type -> Polars dtype resolution
# ---------------------------------------------------------------------------


def _apply_extra_schema(schema_cls: type[Schema], extra_schema: Mapping[str, Any]) -> pl.Schema:
    result = dict(schema_cls.polars_schema())
    fields_by_alias = {info.alias: info for info in schema_cls.__schema_fields__.values()}

    for name, dtype in extra_schema.items():
        info = fields_by_alias.get(name)
        if info is None:
            if schema_cls.__extras_field__ is None:
                raise TypeError(f"{schema_cls.__name__} does not declare Extras for {name!r}")
            if isinstance(dtype, Mapping):
                raise TypeError(f"Extra column {name!r} requires a Polars dtype")
            result[name] = dtype
            continue

        if not isinstance(dtype, Mapping):
            raise TypeError(f"Extra schema conflicts with declared field {name!r}")

        if isinstance(info, Struct):
            nested = _require_schema_type(info.python_type, info)
            result[name] = pl.Struct(nested.polars_schema(dtype))
        elif isinstance(info, ListStruct):
            nested = _require_schema_type(info.python_type, info)
            result[name] = pl.List(pl.Struct(nested.polars_schema(dtype)))
        else:
            raise TypeError(f"Declared field {name!r} cannot contain extra fields")

    return pl.Schema(result)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_value(value: Any, *, by_alias: bool) -> Any:
    if isinstance(value, Schema):
        return value.to_dict(by_alias=by_alias)
    if isinstance(value, list):
        return [_serialize_value(item, by_alias=by_alias) for item in value]
    if isinstance(value, tuple):
        return tuple(_serialize_value(item, by_alias=by_alias) for item in value)
    return value


def _deserialize_value(annotation: Any, value: Any, *, by_alias: bool) -> Any:
    if value is None:
        return None

    annotation = base_annotation(annotation)
    origin = get_origin(annotation)

    if isinstance(annotation, type) and issubclass(annotation, Schema):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Expected mapping for nested {annotation.__name__}, got {type(value).__name__}"
            )
        return annotation.from_dict(value, by_alias=by_alias)

    if origin is list:
        (item_type,) = get_args(annotation)
        if not isinstance(value, list):
            raise TypeError(f"Expected list, got {type(value).__name__}")
        return [_deserialize_value(item_type, item, by_alias=by_alias) for item in value]

    if origin is dict:
        key_type, value_type = get_args(annotation)
        if isinstance(value, Mapping):
            return {
                _deserialize_value(key_type, key, by_alias=by_alias): _deserialize_value(
                    value_type,
                    item,
                    by_alias=by_alias,
                )
                for key, item in value.items()
            }
        if not isinstance(value, list):
            raise TypeError(f"Expected mapping entries, got {type(value).__name__}")

        result: dict[Any, Any] = {}
        for entry in value:
            if not isinstance(entry, Mapping) or "key" not in entry or "value" not in entry:
                raise TypeError("Expected dict entry with 'key' and 'value' fields")
            key = _deserialize_value(key_type, entry["key"], by_alias=by_alias)
            result[key] = _deserialize_value(
                value_type,
                entry["value"],
                by_alias=by_alias,
            )
        return result

    return value


# ---------------------------------------------------------------------------
# Bound nested column construction
# ---------------------------------------------------------------------------


def _build_expr(root_alias: str, steps: tuple[tuple[str, str], ...]) -> pl.Expr:
    def apply(expr: pl.Expr, remaining: tuple[tuple[str, str], ...]) -> pl.Expr:
        if not remaining:
            return expr

        kind, alias = remaining[0]
        tail = remaining[1:]

        if kind == "struct":
            return apply(expr.struct.field(alias), tail)

        if kind == "list_item":
            inner = apply(pl.element().struct.field(alias), tail)
            return expr.list.eval(inner)

        raise RuntimeError(f"Unknown column path step: {kind!r}")

    return apply(pl.col(root_alias), steps)


def _build_column(
    info: _FieldBase,
    *,
    root_alias: str,
    steps: tuple[tuple[str, str], ...],
    python_path: tuple[str, ...],
    polars_path: tuple[str, ...],
    flat_path: tuple[str, ...],
) -> Column[Any]:
    kwargs = {
        "name": info.name,
        "alias": info.alias,
        "dtype": info.polars_dtype,
        "root_alias": root_alias,
        "steps": steps,
        "python_path": python_path,
        "polars_path": polars_path,
        "flat_path": flat_path,
        "field_info": info,
    }

    if isinstance(info, Struct):
        nested_schema = _require_schema_type(info.python_type, info)
        children = _build_bound_children(
            nested_schema,
            root_alias=root_alias,
            parent_steps=steps,
            edge_kind="struct",
            parent_python_path=python_path,
            parent_polars_path=polars_path,
            parent_flat_path=flat_path,
        )
        return StructColumn(fields_proxy=_BoundColumnsProxy(children), **kwargs)

    if isinstance(info, ListStruct):
        nested_schema = _require_schema_type(info.python_type, info)
        children = _build_bound_children(
            nested_schema,
            root_alias=root_alias,
            parent_steps=steps,
            edge_kind="list_item",
            parent_python_path=python_path + ("item",),
            parent_polars_path=polars_path + ("[]",),
            parent_flat_path=flat_path,
        )
        return ListStructColumn(item_proxy=_BoundColumnsProxy(children), **kwargs)

    return Column(**kwargs)


def _require_schema_type(value: Any, info: _FieldBase) -> type[Schema]:
    value = base_annotation(value)
    if not isinstance(value, type) or not issubclass(value, Schema):
        raise TypeError(
            f"{type(info).__name__}[...] for {info.name!r} requires a Schema subclass, "
            f"got {value!r}"
        )
    return value


def _build_bound_children(
    schema_cls: type[Schema],
    *,
    root_alias: str,
    parent_steps: tuple[tuple[str, str], ...],
    edge_kind: str,
    parent_python_path: tuple[str, ...],
    parent_polars_path: tuple[str, ...],
    parent_flat_path: tuple[str, ...],
) -> dict[str, Column[Any]]:
    result: dict[str, Column[Any]] = {}

    for child in schema_cls.__schema_fields__.values():
        result[child.name] = _build_column(
            child,
            root_alias=root_alias,
            steps=parent_steps + ((edge_kind, child.alias),),
            python_path=parent_python_path + (child.name,),
            polars_path=parent_polars_path + (child.alias,),
            flat_path=parent_flat_path
            + (
                child.flat_alias
                if isinstance(child, (Struct, ListStruct)) and child.flat_alias
                else child.alias,
            ),
        )

    return result

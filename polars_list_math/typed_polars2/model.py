"""Slots-dataclass models backed by tuple-oriented Polars construction."""

from __future__ import annotations

import keyword
from collections import namedtuple
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from dataclasses import field as dataclass_field
from types import UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Self,
    Union,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
)

import polars as pl

from polars_list_math.typed_polars.dtypes import annotation_to_dtype

_METADATA_KEY = "polars_list_math.typed_polars2"
_UNSET = object()
_KeyValuePhysical = namedtuple("_KeyValuePhysical", ("key", "value"))


@dataclass(frozen=True, slots=True)
class _FieldOptions:
    alias: str | None = None
    dtype: Any | None = None
    flat: bool = False
    flat_divider: str = ":"
    extras: bool = False


def field[T](
    *,
    default: T | object = _UNSET,
    default_factory: Callable[[], T] | object = _UNSET,
    polars_name: str | None = None,
    dtype: Any | None = None,
    flat: bool = False,
    flat_divider: str = ":",
    repr: bool = True,  # noqa: A002
) -> T:
    """Declare dataclass defaults and Polars-specific field metadata."""
    options = _FieldOptions(polars_name, dtype, flat, flat_divider)
    kwargs: dict[str, Any] = {"metadata": {_METADATA_KEY: options}, "repr": repr}
    if default is not _UNSET:
        kwargs["default"] = default
    if default_factory is not _UNSET:
        kwargs["default_factory"] = default_factory
    return cast(T, dataclass_field(**kwargs))


type Extras = dict[str, Any]


def extras(*, default_factory: Callable[[], Extras] = dict) -> Extras:
    """Declare dynamic fields that become additional physical columns."""
    options = _FieldOptions(extras=True)
    return cast(
        Extras,
        dataclass_field(
            default_factory=default_factory,
            metadata={_METADATA_KEY: options},
            repr=False,
        ),
    )


@dataclass(frozen=True, slots=True)
class Column[T]:
    """Typed Polars column path exposed through ``Model.columns``."""

    name: str
    alias: str
    dtype: Any
    root_alias: str
    steps: tuple[tuple[str, str], ...] = ()

    def expr(self) -> pl.Expr:
        expr = pl.col(self.root_alias)
        for kind, name in self.steps:
            if kind == "struct":
                expr = expr.struct.field(name)
            elif kind == "list_struct":
                expr = expr.list.eval(pl.element().struct.field(name))
            else:  # pragma: no cover - plans only emit the two known steps
                raise RuntimeError(f"Unknown column path step: {kind}")
        return expr


class _Columns:
    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name: str) -> Any:
        return self._values[name]


@dataclass(frozen=True, slots=True)
class StructColumn[T](Column[T]):
    fields: Any = None


@dataclass(frozen=True, slots=True)
class ListStructColumn[T](Column[list[T]]):
    item: Any = None


@dataclass(frozen=True, slots=True)
class _FieldPlan:
    name: str
    alias: str
    annotation: Any
    dtype: Any
    kind: str
    nested: type[Model] | None
    flat: bool
    flat_divider: str


@dataclass(frozen=True, slots=True)
class _ModelPlan:
    model: type[Model]
    fields: tuple[_FieldPlan, ...]
    extras_name: str | None


@dataclass(frozen=True, slots=True)
class _PhysicalField:
    name: str
    dtype: Any
    getter: Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class _PhysicalPlan:
    fields: tuple[_PhysicalField, ...]
    row_type: type[tuple[Any, ...]] | None

    @property
    def schema(self) -> pl.Schema:
        return pl.Schema({item.name: item.dtype for item in self.fields})

    def serialize(self, value: Any) -> tuple[Any, ...]:
        values = tuple(item.getter(value) for item in self.fields)
        return self.row_type(*values) if self.row_type is not None else values


class Model:
    """Base class for models decorated with :func:`model`."""

    __tp2_plan__: ClassVar[_ModelPlan]
    columns: ClassVar[Any]

    @classmethod
    def polars_schema(cls) -> pl.Schema:
        return _build_physical_plan(cls, []).schema

    @classmethod
    def to_frame_many(cls, rows: Iterable[Self], *, strict: bool = True) -> pl.DataFrame:
        materialized = list(rows)
        plan = _build_physical_plan(cls, materialized, top_level=True)
        physical_rows = [plan.serialize(row) for row in materialized]
        return pl.DataFrame(physical_rows, orient="row", schema=plan.schema, strict=strict)

    def to_frame(self, *, strict: bool = True) -> pl.DataFrame:
        return type(self).to_frame_many([self], strict=strict)

    def to_dict(self, *, by_alias: bool = False) -> dict[str, Any]:
        plan = type(self).__tp2_plan__
        result: dict[str, Any] = {}
        for item in plan.fields:
            value = getattr(self, item.name)
            key = item.alias if by_alias else item.name
            if item.kind == "struct" and value is not None:
                value = value.to_dict(by_alias=by_alias)
            elif item.kind == "list_struct" and value is not None:
                value = [child.to_dict(by_alias=by_alias) for child in value]
            result[key] = value
        if plan.extras_name is not None:
            result.update(getattr(self, plan.extras_name))
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, by_alias: bool = False) -> Self:
        plan = cls.__tp2_plan__
        remaining = dict(data)
        kwargs: dict[str, Any] = {}
        for item in plan.fields:
            key = item.alias if by_alias else item.name
            if key not in remaining:
                continue
            value = remaining.pop(key)
            if item.kind == "struct" and value is not None:
                assert item.nested is not None
                value = item.nested.from_dict(value, by_alias=by_alias)
            elif item.kind == "list_struct" and value is not None:
                assert item.nested is not None
                value = [item.nested.from_dict(child, by_alias=by_alias) for child in value]
            elif item.kind == "dict" and isinstance(value, list):
                value = {entry["key"]: entry["value"] for entry in value}
            kwargs[item.name] = value
        if plan.extras_name is not None:
            kwargs[plan.extras_name] = {
                name: value for name, value in remaining.items() if value is not None
            }
        elif remaining:
            raise TypeError(f"Unexpected key(s) for {cls.__name__}: {', '.join(sorted(remaining))}")
        return cls(**kwargs)

    @classmethod
    def iter_frame(cls, frame: pl.DataFrame) -> Iterable[Self]:
        for row in frame.iter_rows(named=True):
            yield cls.from_dict(_unflatten(cls, row), by_alias=True)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> Self:
        if frame.height != 1:
            raise ValueError(f"Expected exactly one row, got {frame.height}")
        return next(iter(cls.iter_frame(frame)))


@dataclass_transform(field_specifiers=(field, extras))
def model[T: type[Model]](cls: T) -> T:
    """Compile a class into a slots-dataclass Polars model."""
    if not is_dataclass(cls):
        cls = cast(T, dataclass(slots=True)(cls))
    elif "__slots__" not in cls.__dict__:
        raise TypeError("@model requires @dataclass(slots=True)")

    hints = get_type_hints(cls, include_extras=True)
    plans: list[_FieldPlan] = []
    extras_name: str | None = None
    aliases: set[str] = set()
    for item in fields(cast(Any, cls)):
        annotation = hints[item.name]
        options = cast(_FieldOptions, item.metadata.get(_METADATA_KEY, _FieldOptions()))
        if options.extras:
            if extras_name is not None:
                raise TypeError(f"{cls.__name__} declares more than one extras field")
            extras_name = item.name
            continue
        alias = options.alias or item.name
        _validate_alias(alias)
        if alias in aliases:
            raise TypeError(f"{cls.__name__} has duplicate alias {alias!r}")
        aliases.add(alias)
        base = _base_annotation(annotation)
        origin = get_origin(base)
        nested: type[Model] | None = None
        kind = "scalar"
        if isinstance(base, type) and issubclass(base, Model):
            _require_model(base)
            nested, kind = base, "struct"
        elif origin is list:
            (inner,) = get_args(base)
            inner = _base_annotation(inner)
            if isinstance(inner, type) and issubclass(inner, Model):
                _require_model(inner)
                nested, kind = inner, "list_struct"
        elif origin is dict:
            kind = "dict"
        dtype = options.dtype
        if dtype is None and kind in ("scalar", "dict"):
            dtype = annotation_to_dtype(annotation)
        plans.append(
            _FieldPlan(
                item.name,
                alias,
                annotation,
                dtype,
                kind,
                nested,
                options.flat,
                options.flat_divider,
            )
        )
    type.__setattr__(cls, "__tp2_plan__", _ModelPlan(cls, tuple(plans), extras_name))
    type.__setattr__(cls, "columns", _build_columns(cls))
    return cls


def _build_physical_plan(
    cls: type[Model], rows: list[Any], *, top_level: bool = False
) -> _PhysicalPlan:
    logical = cls.__tp2_plan__
    physical: list[_PhysicalField] = []
    for item in logical.fields:
        if item.kind in ("scalar", "dict"):
            getter = (
                _dict_getter(item.name)
                if item.kind == "dict"
                else lambda row, name=item.name: getattr(row, name)
            )
            physical.append(
                _PhysicalField(
                    item.alias,
                    item.dtype,
                    getter,
                )
            )
            continue
        assert item.nested is not None
        nested_values = _nested_values(rows, item)
        nested_plan = _build_physical_plan(item.nested, nested_values)
        if not item.flat:
            dtype = pl.Struct(nested_plan.schema)
            if item.kind == "list_struct":
                dtype = pl.List(dtype)
                getter = _list_struct_getter(item.name, nested_plan)
            else:
                getter = _struct_getter(item.name, nested_plan)
            physical.append(_PhysicalField(item.alias, dtype, getter))
            continue
        for index, child in enumerate(nested_plan.fields):
            name = f"{item.alias}{item.flat_divider}{child.name}"
            dtype = child.dtype if item.kind == "struct" else pl.List(child.dtype)
            getter = _flat_getter(item, nested_plan, index)
            physical.append(_PhysicalField(name, dtype, getter))

    if logical.extras_name is not None:
        extra_names = sorted(
            {
                key
                for row in rows
                for key in cast(Mapping[str, Any], getattr(row, logical.extras_name))
            }
        )
        declared = {item.name for item in physical}
        for name in extra_names:
            _validate_alias(name)
            if name in declared:
                raise TypeError(f"Extra field conflicts with declared field {name!r}")
            values = [getattr(row, logical.extras_name).get(name) for row in rows]
            dtype = pl.Series(name, values).dtype
            physical.append(
                _PhysicalField(
                    name,
                    dtype,
                    lambda row, field_name=name, extras_name=logical.extras_name: getattr(
                        row, extras_name
                    ).get(field_name),
                )
            )

    row_type = None if top_level else _make_namedtuple(cls.__name__, physical)
    return _PhysicalPlan(tuple(physical), row_type)


def _make_namedtuple(name: str, physical: list[_PhysicalField]) -> type[tuple[Any, ...]]:
    return cast(type[tuple[Any, ...]], namedtuple(f"_{name}Physical", [x.name for x in physical]))


def _nested_values(rows: list[Any], item: _FieldPlan) -> list[Any]:
    values = [getattr(row, item.name) for row in rows]
    if item.kind == "struct":
        return [value for value in values if value is not None]
    return [child for value in values if value is not None for child in value]


def _struct_getter(name: str, plan: _PhysicalPlan) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, name)
        return None if value is None else plan.serialize(value)

    return get


def _dict_getter(name: str) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, name)
        return (
            None if value is None else [_KeyValuePhysical(key, item) for key, item in value.items()]
        )

    return get


def _list_struct_getter(name: str, plan: _PhysicalPlan) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, name)
        return None if value is None else [plan.serialize(child) for child in value]

    return get


def _flat_getter(item: _FieldPlan, plan: _PhysicalPlan, index: int) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, item.name)
        if value is None:
            return None
        if item.kind == "struct":
            return plan.serialize(value)[index]
        return [plan.serialize(child)[index] for child in value]

    return get


def _build_columns(
    cls: type[Model],
    root: str | None = None,
    steps: tuple[tuple[str, str], ...] = (),
    container_kind: str = "struct",
) -> _Columns:
    values: dict[str, Any] = {}
    for item in cls.__tp2_plan__.fields:
        root_alias = root or item.alias
        item_steps = steps + ((container_kind, item.alias),) if root is not None else ()
        if item.kind == "struct" and not item.flat:
            nested = _build_columns(
                cast(type[Model], item.nested), root_alias, item_steps, "struct"
            )
            values[item.name] = StructColumn(
                item.name, item.alias, None, root_alias, item_steps, nested
            )
        elif item.kind == "list_struct" and not item.flat:
            nested = _build_columns(
                cast(type[Model], item.nested), root_alias, item_steps, "list_struct"
            )
            values[item.name] = ListStructColumn(
                item.name, item.alias, None, root_alias, item_steps, nested
            )
        else:
            values[item.name] = Column(item.name, item.alias, item.dtype, root_alias, item_steps)
    return _Columns(values)


def _unflatten(cls: type[Model], data: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(data)
    for item in cls.__tp2_plan__.fields:
        if not item.flat:
            continue
        prefix = f"{item.alias}{item.flat_divider}"
        selected = {
            key[len(prefix) :]: result.pop(key) for key in tuple(result) if key.startswith(prefix)
        }
        if item.kind == "struct":
            result[item.alias] = (
                None if selected and all(value is None for value in selected.values()) else selected
            )
        else:
            lengths = {len(value) for value in selected.values() if value is not None}
            result[item.alias] = (
                None
                if not lengths
                else [
                    {name: value[index] for name, value in selected.items()}
                    for index in range(next(iter(lengths)))
                ]
            )
    return result


def _base_annotation(annotation: Any) -> Any:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [item for item in get_args(annotation) if item is not type(None)]
        if len(args) == 1:
            return _base_annotation(args[0])
    return annotation


def _validate_alias(alias: str) -> None:
    if not alias.isidentifier() or keyword.iskeyword(alias) or alias.startswith("_"):
        raise TypeError(f"Alias {alias!r} cannot be represented by NamedTuple")


def _require_model(cls: type[Model]) -> None:
    if not hasattr(cls, "__tp2_plan__"):
        raise TypeError(f"Nested model {cls.__name__} must be decorated with @model")

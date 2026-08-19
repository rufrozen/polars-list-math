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
    if not isinstance(flat, bool):
        raise TypeError("flat must be a bool")
    if not isinstance(flat_divider, str) or not flat_divider:
        raise TypeError("flat_divider must be a non-empty string")
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
        return _apply_column_steps(pl.col(self.root_alias), self.steps)


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

    __slots__ = ()

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

    def to_dict(self, *, by_polars_name: bool = False) -> dict[str, Any]:
        """Serialize this model using Python or physical Polars field names."""
        plan = type(self).__tp2_plan__
        result: dict[str, Any] = {}
        for item in plan.fields:
            value = getattr(self, item.name)
            key = item.alias if by_polars_name else item.name
            if item.kind == "struct" and value is not None:
                value = value.to_dict(by_polars_name=by_polars_name)
            elif item.kind == "list_struct" and value is not None:
                value = [child.to_dict(by_polars_name=by_polars_name) for child in value]
            result[key] = value
        if plan.extras_name is not None:
            result.update(getattr(self, plan.extras_name))
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, by_polars_name: bool = False) -> Self:
        """Build a model from Python or physical Polars field names."""
        plan = cls.__tp2_plan__
        remaining = dict(data)
        kwargs: dict[str, Any] = {}
        for item in plan.fields:
            key = item.alias if by_polars_name else item.name
            if key not in remaining:
                continue
            value = remaining.pop(key)
            if item.kind == "struct" and value is not None:
                assert item.nested is not None
                if not isinstance(value, Mapping):
                    raise TypeError(f"Field {item.name!r} must be a mapping")
                value = item.nested.from_dict(value, by_polars_name=by_polars_name)
            elif item.kind == "list_struct" and value is not None:
                assert item.nested is not None
                if not isinstance(value, list) or not all(
                    isinstance(child, Mapping) for child in value
                ):
                    raise TypeError(f"Field {item.name!r} must be a list of mappings")
                value = [
                    item.nested.from_dict(child, by_polars_name=by_polars_name) for child in value
                ]
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
    def iter_frame(cls, frame: pl.DataFrame, *, strict_schema: bool = False) -> Iterable[Self]:
        if strict_schema:
            cls.assert_frame_schema(frame)
        for row in frame.iter_rows(named=True):
            yield cls.from_dict(_unflatten(cls, row), by_polars_name=True)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame, *, strict_schema: bool = False) -> Self:
        if frame.height != 1:
            raise ValueError(f"Expected exactly one row, got {frame.height}")
        return next(iter(cls.iter_frame(frame, strict_schema=strict_schema)))

    @classmethod
    def assert_frame_schema(cls, frame: pl.DataFrame) -> None:
        """Require the exact declared schema for models without dynamic extras."""
        if _model_has_extras(cls):
            raise TypeError("Strict schema validation is unavailable for models with Extras")
        expected = cls.polars_schema()
        if frame.schema != expected:
            raise TypeError(f"Unexpected DataFrame schema: expected {expected}, got {frame.schema}")


@dataclass_transform(field_specifiers=(field, extras))
def model[T: type[Model]](cls: T) -> T:
    """Compile a class into a slots-dataclass Polars model."""
    if not is_dataclass(cls) or "__dataclass_fields__" not in cls.__dict__:
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
        if options.flat and kind not in ("struct", "list_struct"):
            raise TypeError(
                f"Field {cls.__name__}.{item.name} cannot use flat=True; "
                "only nested models and lists of nested models can be flat"
            )
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

    physical_names: set[str] = set()
    for item in physical:
        if item.name in physical_names:
            raise TypeError(f"Physical field name conflict for {item.name!r}")
        physical_names.add(item.name)

    row_type = None if top_level else _make_namedtuple(cls.__name__, physical)
    return _PhysicalPlan(tuple(physical), row_type)


def _make_namedtuple(name: str, physical: list[_PhysicalField]) -> type[tuple[Any, ...]]:
    return cast(type[tuple[Any, ...]], namedtuple(f"_{name}Physical", [x.name for x in physical]))


def _nested_values(rows: list[Any], item: _FieldPlan) -> list[Any]:
    values = [getattr(row, item.name) for row in rows]
    assert item.nested is not None
    if item.kind == "struct":
        invalid = next(
            (value for value in values if value is not None and not isinstance(value, item.nested)),
            None,
        )
        if invalid is not None:
            raise TypeError(
                f"Field {item.name!r} expected {item.nested.__name__}, got {type(invalid).__name__}"
            )
        return [value for value in values if value is not None]
    for value in values:
        if value is not None and (
            not isinstance(value, list)
            or any(not isinstance(child, item.nested) for child in value)
        ):
            raise TypeError(f"Field {item.name!r} must be a list of {item.nested.__name__}")
    return [child for value in values if value is not None for child in value]


def _struct_getter(name: str, plan: _PhysicalPlan) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, name)
        return None if value is None else plan.serialize(value)

    return get


def _dict_getter(name: str) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, name)
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"Field {name!r} must be a mapping")
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
        elif item.flat:
            nested = _build_flat_columns(
                cast(type[Model], item.nested),
                prefix=item.alias,
                divider=item.flat_divider,
                as_list=item.kind == "list_struct",
            )
            if item.kind == "struct":
                values[item.name] = StructColumn(
                    item.name, item.alias, None, item.alias, (), nested
                )
            else:
                values[item.name] = ListStructColumn(
                    item.name, item.alias, None, item.alias, (), nested
                )
        else:
            values[item.name] = Column(item.name, item.alias, item.dtype, root_alias, item_steps)
    return _Columns(values)


def _build_flat_columns(cls: type[Model], *, prefix: str, divider: str, as_list: bool) -> _Columns:
    values: dict[str, Any] = {}
    for item in cls.__tp2_plan__.fields:
        physical_name = f"{prefix}{divider}{item.alias}"
        if item.kind == "struct" and not item.flat:
            nested = _build_columns(cast(type[Model], item.nested), physical_name)
            values[item.name] = StructColumn(item.name, item.alias, None, physical_name, (), nested)
        elif item.kind == "list_struct" and not item.flat:
            nested = _build_columns(cast(type[Model], item.nested), physical_name)
            values[item.name] = ListStructColumn(
                item.name, item.alias, None, physical_name, (), nested
            )
        elif item.flat:
            values[item.name] = _build_flat_columns(
                cast(type[Model], item.nested),
                prefix=physical_name,
                divider=item.flat_divider,
                as_list=as_list or item.kind == "list_struct",
            )
        else:
            dtype = pl.List(item.dtype) if as_list else item.dtype
            values[item.name] = Column(item.name, item.alias, dtype, physical_name, ())
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
            if len(lengths) > 1:
                raise TypeError(
                    f"Flat ListStruct columns for {item.alias!r} have different lengths"
                )
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
        raise TypeError(f"Polars name {alias!r} cannot be represented by NamedTuple")


def _require_model(cls: type[Model]) -> None:
    if not hasattr(cls, "__tp2_plan__"):
        raise TypeError(f"Nested model {cls.__name__} must be decorated with @model")


def _apply_column_steps(expr: pl.Expr, steps: tuple[tuple[str, str], ...]) -> pl.Expr:
    if not steps:
        return expr
    (kind, name), remaining = steps[0], steps[1:]
    if kind == "struct":
        return _apply_column_steps(expr.struct.field(name), remaining)
    if kind == "list_struct":
        nested = _apply_column_steps(pl.element().struct.field(name), remaining)
        return expr.list.eval(nested)
    raise RuntimeError(f"Unknown column path step: {kind}")


def _model_has_extras(cls: type[Model]) -> bool:
    plan = cls.__tp2_plan__
    if plan.extras_name is not None:
        return True
    return any(item.nested is not None and _model_has_extras(item.nested) for item in plan.fields)

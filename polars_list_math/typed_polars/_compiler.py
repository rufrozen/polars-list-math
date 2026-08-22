"""Recursive model validation and physical plan compilation."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable, Mapping
from dataclasses import MISSING, fields, is_dataclass
from types import UnionType
from typing import Annotated, Any, TypeAliasType, Union, cast, get_args, get_origin, get_type_hints

import polars as pl

from ._plans import FieldPlan, ModelPlan, PhysicalField, PhysicalPlan
from .context import Context, context_keys
from .schema import FlatDict, ListStruct, Schema, Struct


def compile_model[T: type](cls: T, schema: type[Schema], *, strict: bool) -> T:
    """Recursively validate a dataclass tree and attach its logical plans."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")

    hints = get_type_hints(cls, include_extras=True)
    plans: list[FieldPlan] = []
    for item in fields(cast(Any, cls)):
        annotation = hints[item.name]
        base = _base_annotation(annotation)
        origin = get_origin(base)
        nested: type[Any] | None = None
        kind = "scalar"
        column = schema.fields().get(item.name)
        if isinstance(base, type) and is_dataclass(base):
            if column is not None and isinstance(column, Struct):
                _prepare_model(base, column.schema, strict=strict)
            nested, kind = base, "struct"
        elif origin is list:
            (inner,) = get_args(base)
            inner = _base_annotation(inner)
            if isinstance(inner, type) and is_dataclass(inner):
                if column is not None and isinstance(column, ListStruct):
                    _prepare_model(inner, column.schema, strict=strict)
                nested, kind = inner, "list_struct"
        elif origin is dict:
            kind = "flat_dict" if isinstance(column, FlatDict) else "dict"
        has_default = item.default is not MISSING or item.default_factory is not MISSING
        plans.append(FieldPlan(item.name, annotation, kind, nested, has_default))

    type.__setattr__(cls, "__tp2_plan__", ModelPlan(cls, tuple(plans)))
    type.__setattr__(cls, "__tp_schema__", schema)
    type.__setattr__(cls, "__tp_schema_strict__", strict)
    validate_model_schema(cls, schema, strict=strict)
    return cast(T, cls)


def validate_model_schema(
    model_cls: type[Any], schema: type[Schema], *, strict: bool = False
) -> None:
    """Validate compatible model/schema fields recursively."""
    plan = getattr(model_cls, "__tp2_plan__", None)
    if plan is None:
        raise TypeError(f"{model_cls!r} is not a compiled typed Polars model")

    available = schema.fields()
    for item in plan.fields:
        column = available.get(item.name)
        if column is None:
            if strict:
                raise TypeError(
                    f"Model field {model_cls.__name__}.{item.name} is not declared "
                    f"in schema {schema.__name__}"
                )
            if not item.has_default:
                raise TypeError(
                    f"Model field {model_cls.__name__}.{item.name} is not declared in "
                    f"schema {schema.__name__} and must define a default"
                )
            continue

        if item.kind == "struct":
            if not isinstance(column, Struct) or item.nested is None:
                raise TypeError(f"Model field {item.name!r} must match a Struct schema field")
            validate_model_schema(item.nested, column.schema, strict=strict)
        elif item.kind == "list_struct":
            if not isinstance(column, ListStruct) or item.nested is None:
                raise TypeError(f"Model field {item.name!r} must match a ListStruct schema field")
            validate_model_schema(item.nested, column.schema, strict=strict)
        elif isinstance(column, (Struct, ListStruct)):
            raise TypeError(f"Model field {item.name!r} does not match its schema structure")
        elif not _annotations_match(item.annotation, column.python_type):
            raise TypeError(
                f"Model field {model_cls.__name__}.{item.name} uses Python type "
                f"{item.annotation!r}, expected {column.python_type!r}"
            )

    if strict:
        model_fields = {item.name for item in plan.fields}
        missing = available.keys() - model_fields
        if missing:
            raise TypeError(
                f"Schema {schema.__name__} field(s) are not declared in model "
                f"{model_cls.__name__}: {', '.join(sorted(missing))}"
            )


def build_physical_plan(
    cls: type[Any],
    schema: type[Schema],
    *,
    context: Context | None = None,
    context_path: tuple[object, ...] = (),
    top_level: bool = False,
) -> PhysicalPlan:
    """Compile the cached tuple-oriented physical serialization plan."""
    logical = cls.__tp2_plan__
    physical: list[PhysicalField] = []
    for item in logical.fields:
        column = schema.fields().get(item.name)
        if column is None:
            continue
        if item.kind == "flat_dict":
            assert isinstance(column, FlatDict)
            for key in context_keys(
                context,
                column,
                context_path + (column,),
            ):
                physical.append(
                    PhysicalField(
                        column.physical_name(key),
                        column.dtype,
                        _flat_dict_getter(item.name, key),
                    )
                )
            continue
        if item.kind in ("scalar", "dict"):
            getter = (
                _dict_getter(item.name)
                if item.kind == "dict"
                else lambda row, name=item.name: getattr(row, name)
            )
            physical.append(PhysicalField(column.polars_name, column.dtype, getter))
            continue

        assert item.nested is not None
        nested_schema = cast(Any, column).schema
        nested_plan = build_physical_plan(
            item.nested,
            nested_schema,
            context=context,
            context_path=context_path + (column,),
        )
        if cast(Any, column).flat:
            for index, child in enumerate(nested_plan.fields):
                name = f"{column.polars_name}{cast(Any, column).divider}{child.name}"
                dtype = child.dtype if item.kind == "struct" else pl.List(child.dtype)
                physical.append(PhysicalField(name, dtype, _flat_getter(item, nested_plan, index)))
            continue
        dtype = pl.Struct(nested_plan.schema)
        if item.kind == "list_struct":
            dtype = pl.List(dtype)
            getter = _list_struct_getter(item.name, nested_plan)
        else:
            getter = _struct_getter(item.name, nested_plan)
        physical.append(PhysicalField(column.polars_name, dtype, getter))

    names: set[str] = set()
    for item in physical:
        if item.name in names:
            raise TypeError(f"Physical field name conflict for {item.name!r}")
        names.add(item.name)
    row_type = None if top_level else _make_namedtuple(cls.__name__, physical)
    return PhysicalPlan(tuple(physical), row_type)


_KeyValuePhysical = namedtuple("_KeyValuePhysical", ("key", "value"))


def _make_namedtuple(name: str, physical: list[PhysicalField]) -> type[tuple[Any, ...]]:
    return cast(type[tuple[Any, ...]], namedtuple(f"_{name}Physical", [x.name for x in physical]))


def _struct_getter(name: str, plan: PhysicalPlan) -> Callable[[Any], Any]:
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


def _flat_dict_getter(name: str, key: str) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, name)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(f"Field {name!r} must be a mapping")
        return value.get(key)

    return get


def _list_struct_getter(name: str, plan: PhysicalPlan) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, name)
        return None if value is None else [plan.serialize(child) for child in value]

    return get


def _flat_getter(item: FieldPlan, plan: PhysicalPlan, index: int) -> Callable[[Any], Any]:
    def get(row: Any) -> Any:
        value = getattr(row, item.name)
        if value is None:
            return None
        if item.kind == "struct":
            return plan.serialize(value)[index]
        return [plan.serialize(child)[index] for child in value]

    return get


def _prepare_model[T: type](cls: T, schema: type[Schema], *, strict: bool) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} must be a dataclass")
    if (
        cls.__dict__.get("__tp_schema__") is schema
        and cls.__dict__.get("__tp_schema_strict__") is strict
        and "__tp2_plan__" in cls.__dict__
    ):
        return cast(T, cls)
    return cast(T, compile_model(cls, schema, strict=strict))


def _base_annotation(annotation: Any) -> Any:
    while isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [item for item in get_args(annotation) if item is not type(None)]
        if len(args) == 1:
            return _base_annotation(args[0])
    return annotation


def _annotations_match(left: Any, right: Any) -> bool:
    left = _logical_annotation(left)
    right = _logical_annotation(right)
    left_origin = get_origin(left)
    right_origin = get_origin(right)
    if left_origin != right_origin:
        return False
    if left_origin is None:
        return left == right
    left_args = get_args(left)
    right_args = get_args(right)
    return len(left_args) == len(right_args) and all(
        _annotations_match(left_item, right_item)
        for left_item, right_item in zip(left_args, right_args, strict=True)
    )


def _logical_annotation(annotation: Any) -> Any:
    while isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
        while isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        args = [item for item in get_args(annotation) if item is not type(None)]
        if len(args) == 1:
            return _logical_annotation(args[0])
    return annotation

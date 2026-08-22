"""Python annotations and their physical Polars data types."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import UnionType
from typing import Annotated, Any, TypeAliasType, Union, get_args, get_origin

import polars as pl

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
    "DurationMs",
    "DurationNs",
    "DurationUs",
    "TimestampMs",
    "TimestampNs",
    "TimestampUs",
]

type I8 = Annotated[int, pl.Int8]
type I16 = Annotated[int, pl.Int16]
type I32 = Annotated[int, pl.Int32]
type I64 = Annotated[int, pl.Int64]
type U8 = Annotated[int, pl.UInt8]
type U16 = Annotated[int, pl.UInt16]
type U32 = Annotated[int, pl.UInt32]
type U64 = Annotated[int, pl.UInt64]
type F32 = Annotated[float, pl.Float32]
type F64 = Annotated[float, pl.Float64]

type TimestampMs = Annotated[datetime, pl.Datetime("ms")]
type TimestampUs = Annotated[datetime, pl.Datetime("us")]
type TimestampNs = Annotated[datetime, pl.Datetime("ns")]
type DurationMs = Annotated[timedelta, pl.Duration("ms")]
type DurationUs = Annotated[timedelta, pl.Duration("us")]
type DurationNs = Annotated[timedelta, pl.Duration("ns")]


_DEFAULT_DTYPES: dict[Any, Any] = {
    str: pl.String,
    int: pl.Int64,
    float: pl.Float64,
    bool: pl.Boolean,
    bytes: pl.Binary,
    date: pl.Date,
    datetime: pl.Datetime("us"),
    timedelta: pl.Duration("us"),
}


def annotation_to_dtype(annotation: Any) -> Any:
    annotation, annotated_dtype = _strip_annotated(annotation)
    if annotated_dtype is not None:
        return annotated_dtype

    annotation = _strip_optional(annotation)
    annotation, annotated_dtype = _strip_annotated(annotation)
    if annotated_dtype is not None:
        return annotated_dtype

    origin = get_origin(annotation)
    if origin is list:
        (item_type,) = get_args(annotation)
        return pl.List(annotation_to_dtype(item_type))

    if origin is dict:
        key_type, value_type = get_args(annotation)
        return pl.List(
            pl.Struct(
                {
                    "key": annotation_to_dtype(key_type),
                    "value": annotation_to_dtype(value_type),
                }
            )
        )

    try:
        return _DEFAULT_DTYPES[annotation]
    except KeyError:
        raise TypeError(
            f"Cannot infer a Polars dtype from {annotation!r}. "
            "Use an Annotated alias such as I32/F32 or pass dtype= explicitly."
        ) from None


def base_annotation(annotation: Any) -> Any:
    annotation, _ = _strip_annotated(annotation)
    annotation = _strip_optional(annotation)
    annotation, _ = _strip_annotated(annotation)
    return annotation


def is_nullable(annotation: Any) -> bool:
    """Return whether an annotation explicitly accepts ``None``."""
    annotation, _ = _strip_annotated(annotation)
    annotation = _unwrap_type_alias(annotation)
    origin = get_origin(annotation)
    return origin in (Union, UnionType) and type(None) in get_args(annotation)


def _is_polars_dtype(value: Any) -> bool:
    try:
        if isinstance(value, pl.DataType):
            return True
    except TypeError:
        pass

    try:
        return isinstance(value, type) and issubclass(value, pl.DataType)
    except TypeError:
        return False


def _unwrap_type_alias(annotation: Any) -> Any:
    while isinstance(annotation, TypeAliasType):
        annotation = annotation.__value__
    return annotation


def _strip_annotated(annotation: Any) -> tuple[Any, Any | None]:
    annotation = _unwrap_type_alias(annotation)
    dtype = None
    while get_origin(annotation) is Annotated:
        base, *metadata = get_args(annotation)
        for item in metadata:
            if _is_polars_dtype(item):
                dtype = item
        annotation = _unwrap_type_alias(base)
    return annotation, dtype


def _strip_optional(annotation: Any) -> Any:
    annotation = _unwrap_type_alias(annotation)
    origin = get_origin(annotation)
    if origin not in (Union, UnionType):
        return annotation

    args = tuple(arg for arg in get_args(annotation) if arg is not type(None))
    if len(args) == 1:
        return args[0]
    return annotation

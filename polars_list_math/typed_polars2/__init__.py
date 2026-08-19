"""Tuple-oriented typed Polars models based on slots dataclasses."""

from polars_list_math.typed_polars.dtypes import (
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
)

from .model import Column, Extras, ListStructColumn, Model, StructColumn, extras, field, model

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
    "ListStructColumn",
    "Model",
    "StructColumn",
    "TimestampMs",
    "TimestampNs",
    "TimestampUs",
    "extras",
    "field",
    "model",
]

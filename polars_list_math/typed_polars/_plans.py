"""Immutable compiled plans shared by typed Polars components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class FieldPlan:
    name: str
    annotation: Any
    kind: str
    nested: type[Any] | None
    has_default: bool


@dataclass(frozen=True, slots=True)
class ModelPlan:
    model: type[Any]
    fields: tuple[FieldPlan, ...]


@dataclass(frozen=True, slots=True)
class PhysicalField:
    name: str
    dtype: Any
    getter: Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    fields: tuple[PhysicalField, ...]
    row_type: type[tuple[Any, ...]] | None

    @property
    def schema(self) -> pl.Schema:
        return pl.Schema({item.name: item.dtype for item in self.fields})

    def serialize(self, value: Any) -> tuple[Any, ...]:
        values = tuple(item.getter(value) for item in self.fields)
        return self.row_type(*values) if self.row_type is not None else values

"""Resolve cached builders attached to root model classes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, cast

import polars as pl

from ._plans import PhysicalPlan
from .context import Context


class BuilderProtocol[T](Protocol):
    """Operations exposed by a compiled model/schema builder."""

    schema: type[Any]
    model: type[T]
    strict: bool
    physical_plan: PhysicalPlan

    def physical_plan_for(self, context: Context | None = None) -> PhysicalPlan: ...

    def polars_schema(self, *, context: Context | None = None) -> pl.Schema: ...

    def to_frame(
        self,
        row: T,
        *,
        context: Context | None = None,
        strict: bool = True,
    ) -> pl.DataFrame: ...

    def to_frame_many(
        self,
        rows: Iterable[T],
        *,
        context: Context | None = None,
        strict: bool = True,
    ) -> pl.DataFrame: ...

    def iter_frame(
        self,
        model: type[T],
        frame: pl.DataFrame,
        *,
        strict_schema: bool = False,
    ) -> Iterable[T]: ...

    def from_frame(
        self,
        model: type[T],
        frame: pl.DataFrame,
        *,
        strict_schema: bool = False,
    ) -> T: ...

    def assert_frame_schema(self, frame: pl.DataFrame) -> None: ...

    def to_dict(
        self,
        row: T,
        *,
        by_polars_name: bool = False,
        context: Context | None = None,
    ) -> dict[str, Any]: ...

    def from_dict(
        self,
        model: type[T],
        data: Mapping[str, Any],
        *,
        by_polars_name: bool = False,
    ) -> T: ...


def get_builder[T](model: type[T], *, schema: type[Any]) -> BuilderProtocol[T]:
    """Return the model builder and verify its schema identity."""
    builder = model.__dict__.get("__tp_builder__")
    if builder is None:
        raise TypeError(f"{model.__name__} has no builder; decorate it with @model(schema=...)")
    if builder.schema is not schema:
        raise TypeError(
            f"Model {model.__name__} is bound to schema {builder.schema.__name__}, "
            f"not {schema.__name__}"
        )
    return cast(BuilderProtocol[T], builder)

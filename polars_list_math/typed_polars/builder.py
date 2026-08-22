"""Validated model/schema bindings and cached DataFrame construction plans."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import polars as pl

from ._binding import BuilderProtocol, get_builder
from ._codec import (
    assert_frame_schema,
    from_dict,
    from_frame,
    iter_frame,
    to_dict,
    to_frame_many,
)
from ._compiler import build_physical_plan, compile_model
from ._plans import PhysicalPlan
from .context import Context, context_signature
from .schema import Schema


class Builder[T](BuilderProtocol[T]):
    """Validate a root model against a schema and cache its physical plan."""

    def __init__(
        self,
        schema: type[Schema],
        model: type[T],
        *,
        strict: bool = False,
    ) -> None:
        if not isinstance(schema, type) or not issubclass(schema, Schema):
            raise TypeError("schema must be a typed Polars Schema class")
        if not isinstance(strict, bool):
            raise TypeError("strict must be a bool")

        self.schema = schema
        self.model = compile_model(model, schema, strict=strict)
        self.strict = strict
        self.physical_plan = build_physical_plan(
            self.model,
            schema,
            top_level=True,
        )
        self._physical_plans = {context_signature(None): self.physical_plan}
        type.__setattr__(self.model, "__tp_builder__", self)

    @classmethod
    def for_model(
        cls,
        model: type[T],
        *,
        schema: type[Schema],
    ) -> Builder[T]:
        """Return the cached builder and require the requested schema."""
        return cast(Builder[T], get_builder(model, schema=schema))

    def physical_plan_for(self, context: Context | None = None) -> PhysicalPlan:
        """Return or build the physical plan for a runtime context."""
        if context is None:
            return self.physical_plan
        signature = context_signature(context)
        plan = self._physical_plans.get(signature)
        if plan is None:
            plan = build_physical_plan(
                self.model,
                self.schema,
                context=context,
                top_level=True,
            )
            self._physical_plans[signature] = plan
        return plan

    def polars_schema(self, *, context: Context | None = None) -> pl.Schema:
        """Return the cached physical schema selected by the model."""
        return self.physical_plan_for(context).schema

    def to_frame(
        self,
        row: T,
        *,
        context: Context | None = None,
        strict: bool = True,
    ) -> pl.DataFrame:
        """Serialize one validated root model value."""
        return self.to_frame_many([row], context=context, strict=strict)

    def to_frame_many(
        self,
        rows: Iterable[T],
        *,
        context: Context | None = None,
        strict: bool = True,
    ) -> pl.DataFrame:
        """Serialize validated root model values."""
        return to_frame_many(self.model, rows, context=context, strict=strict)

    def iter_frame(
        self,
        model: type[T],
        frame: pl.DataFrame,
        *,
        strict_schema: bool = False,
    ) -> Iterable[T]:
        """Deserialize DataFrame rows into the explicit root model type."""
        self.validate_model(model)
        return cast(Iterable[T], iter_frame(model, frame, strict_schema=strict_schema))

    def from_frame(
        self,
        model: type[T],
        frame: pl.DataFrame,
        *,
        strict_schema: bool = False,
    ) -> T:
        """Deserialize one DataFrame row into the explicit root model type."""
        self.validate_model(model)
        return cast(T, from_frame(model, frame, strict_schema=strict_schema))

    def assert_frame_schema(self, frame: pl.DataFrame) -> None:
        """Require the cached physical model schema."""
        assert_frame_schema(self.model, frame)

    def to_dict(
        self,
        row: T,
        *,
        by_polars_name: bool = False,
        context: Context | None = None,
    ) -> dict[str, Any]:
        """Serialize one validated model to a dictionary."""
        self.validate_model(type(row))
        return to_dict(
            row,
            by_polars_name=by_polars_name,
            context=context,
        )

    def from_dict(
        self,
        model: type[T],
        data: Mapping[str, Any],
        *,
        by_polars_name: bool = False,
    ) -> T:
        """Deserialize a dictionary into the explicit root model type."""
        self.validate_model(model)
        return cast(
            T,
            from_dict(
                model,
                data,
                by_polars_name=by_polars_name,
                ignore_unknown=False,
            ),
        )

    def validate_model(self, model: type[Any]) -> None:
        """Require the exact root model registered with this builder."""
        if model is not self.model:
            raise TypeError(
                f"Expected model {self.model.__name__} for {self.schema.__name__}, "
                f"got {model.__name__}"
            )

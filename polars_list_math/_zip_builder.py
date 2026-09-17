from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import polars as pl

from . import _list_zip


@dataclass
class _Level:
    inputs: tuple[pl.Expr, ...]
    fields: tuple[str, ...]
    pad: bool
    broadcast: bool
    operations: list[tuple[Literal["eval", "filter"], pl.Expr, bool]] = field(default_factory=list)


class ZipBuilder:
    """Build nested list zips with ordered transformations at each depth.

    The constructor configures the outer zip. Each ``level()`` adds an inner
    zip over the previous level's struct fields. ``eval()`` and ``filter()``
    apply to the current level, in call order, before descending further.
    Use ``pl.element()`` to address an element at that level.

    Fluent methods mutate and return this builder. ``build()`` returns a fresh
    expression without consuming the builder or evaluating any data.
    """

    def __init__(
        self,
        *exprs: Any,
        pad: bool = False,
        broadcast: bool = False,
        fields: Sequence[str] | None = None,
    ) -> None:
        if len(exprs) < 2:
            raise ValueError("ZipBuilder requires at least two expressions")
        inputs = tuple(_list_zip._parse_into_expr(expr) for expr in exprs)
        names = _list_zip._normalize_fields(fields, len(inputs))
        if names is None:
            names = _list_zip._default_field_names(exprs, inputs)
        self._levels = [self._make_level(inputs, names, pad, broadcast)]

    def level(
        self,
        *,
        pad: bool = False,
        broadcast: bool = False,
        fields: Sequence[str] | None = None,
    ) -> Self:
        """Descend one list level, with independent zip options and field names.

        Inputs are the current struct's fields in their original order. An
        earlier eval must preserve those fields if another level follows.
        Omitted field names retain the current names; booleans default to false.
        """
        previous = self._levels[-1]
        inputs = tuple(pl.element().struct.field(name) for name in previous.fields)
        names = previous.fields if fields is None else fields
        self._levels.append(self._make_level(inputs, names, pad, broadcast))
        return self

    def eval(self, expression: pl.Expr, *, parallel: bool = False) -> Self:
        """Transform current-level elements using a Polars list-eval expression."""
        self._validate_expression(expression)
        if not isinstance(parallel, bool):
            raise TypeError("parallel must be a bool")
        self._levels[-1].operations.append(("eval", expression, parallel))
        return self

    def filter(self, predicate: pl.Expr) -> Self:
        """Keep current-level elements whose predicate is true; discard nulls."""
        self._validate_expression(predicate)
        self._levels[-1].operations.append(("filter", predicate, False))
        return self

    def build(self, name: str | None = None) -> pl.Expr:
        """Return the nested expression, optionally aliased to a column name."""
        if name is not None and not isinstance(name, str):
            raise TypeError("name must be a string or None")
        child: pl.Expr | None = None
        for level in reversed(self._levels):
            expression = _list_zip.list_zip(
                *level.inputs,
                pad=level.pad,
                broadcast=level.broadcast,
                fields=level.fields,
            )
            for kind, operation, parallel in level.operations:
                if kind == "filter":
                    expression = expression.list.eval(pl.element().filter(operation))
                else:
                    expression = expression.list.eval(operation, parallel=parallel)
            if child is not None:
                expression = expression.list.eval(child)
            child = expression
        assert child is not None
        # Keep the result name stable across the native and fallback zip paths.
        return child.alias(name if name is not None else self._levels[0].fields[0])

    @staticmethod
    def _make_level(
        inputs: tuple[pl.Expr, ...],
        names: Sequence[str],
        pad: bool,
        broadcast: bool,
    ) -> _Level:
        if not isinstance(pad, bool):
            raise TypeError("pad must be a bool")
        if not isinstance(broadcast, bool):
            raise TypeError("broadcast must be a bool")
        fields = _list_zip._normalize_fields(names, len(inputs))
        assert fields is not None
        return _Level(inputs, tuple(fields), pad, broadcast)

    @staticmethod
    def _validate_expression(expression: pl.Expr) -> None:
        if not isinstance(expression, pl.Expr):
            raise TypeError("expected a Polars expression using pl.element()")
        if any(name != "" for name in expression.meta.root_names()):
            raise ValueError("use pl.element() inside eval/filter, not named columns")

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import Any

import polars as pl
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

IntoExpr = Any

_INPUT_PREFIX = "__polars_list_math_list_zip_input_"
_LIB = Path(__file__).parent
_NATIVE_EXTENSIONS = {".so", ".pyd", ".dll", ".dylib"}


def zip_list(
    *exprs: IntoExpr,
    pad: bool = False,
    fields: Sequence[str] | None = None,
) -> pl.Expr:
    """Zip list expressions into a ``list[struct]`` expression.

    Parameters
    ----------
    *exprs
        List expressions to zip. Strings are parsed as column names.
    pad
        If false, truncate to the shortest list. If true, pad shorter lists with
        nulls to match the longest list.
    fields
        Optional struct field names. Defaults to the input expression names.
    """
    if not isinstance(pad, bool):
        msg = "pad must be a bool"
        raise TypeError(msg)
    if len(exprs) < 2:
        msg = "zip_list requires at least two list expressions"
        raise ValueError(msg)

    field_names = _normalize_fields(fields, len(exprs))
    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="list_zip",
            args=list(exprs),
            kwargs={"pad": pad, "fields": field_names},
            is_elementwise=True,
            use_abs_path=True,
        )

    parsed_exprs = [_parse_into_expr(expr) for expr in exprs]
    fallback_field_names = field_names or _default_field_names(exprs, parsed_exprs)
    aliases = [f"{_INPUT_PREFIX}{i}" for i in range(len(parsed_exprs))]

    row_expr = pl.struct(
        expr.alias(alias) for expr, alias in zip(parsed_exprs, aliases, strict=True)
    )

    def zip_row(row: Any) -> list[dict[str, Any]] | None:
        values = _row_values(row, aliases)
        if any(value is None for value in values):
            return None

        for value in values:
            if not isinstance(value, list):
                msg = "zip_list expects expressions with List or Array dtype"
                raise TypeError(msg)

        rows = zip_longest(*values, fillvalue=None) if pad else zip(*values, strict=False)
        return [dict(zip(fallback_field_names, values_row, strict=True)) for values_row in rows]

    return row_expr.map_elements(
        zip_row,
        return_dtype=None,
        skip_nulls=False,
    )


def install(*, overwrite: bool = False) -> None:
    """Register the Polars list namespace helper."""
    list_namespace = type(pl.col("__polars_list_math_list_zip_probe__").list)

    if overwrite or not hasattr(list_namespace, "zip"):
        list_namespace.zip = _expr_list_zip  # type: ignore


def _expr_list_zip(
    self: Any,
    *others: IntoExpr,
    pad: bool = False,
    fields: Sequence[str] | None = None,
) -> pl.Expr:
    base_expr = wrap_expr(self._pyexpr)
    return zip_list(
        base_expr,
        *others,
        pad=pad,
        fields=fields,
    )


def _parse_into_expr(expr: IntoExpr) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, str):
        return pl.col(expr)
    return pl.lit(expr)


def _normalize_fields(
    fields: Sequence[str] | None,
    width: int,
) -> list[str] | None:
    if fields is None:
        return None
    if isinstance(fields, str):
        msg = "fields must be a sequence of strings, not a single string"
        raise TypeError(msg)

    field_names = list(fields)
    if len(field_names) != width:
        msg = f"fields must contain exactly {width} names"
        raise ValueError(msg)
    if not all(isinstance(field, str) for field in field_names):
        msg = "fields must contain only strings"
        raise TypeError(msg)
    if len(set(field_names)) != len(field_names):
        msg = "fields must be unique"
        raise ValueError(msg)
    return field_names


def _row_values(row: Any, aliases: Sequence[str]) -> list[Any]:
    if row is None:
        return [None] * len(aliases)
    if isinstance(row, Mapping):
        return [row[alias] for alias in aliases]
    return list(row)


def _default_field_names(
    exprs: Sequence[IntoExpr],
    parsed_exprs: Sequence[pl.Expr],
) -> list[str]:
    names: list[str] = []
    for index, (expr, parsed_expr) in enumerate(zip(exprs, parsed_exprs, strict=True)):
        if isinstance(expr, str):
            names.append(expr)
            continue

        try:
            names.append(parsed_expr.meta.output_name())
        except Exception:
            names.append(f"field_{index}")
    return names


def _native_library_available() -> bool:
    return any(path.is_file() and path.suffix in _NATIVE_EXTENSIONS for path in _LIB.iterdir())

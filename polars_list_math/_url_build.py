from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlunparse

import polars as pl
from polars.plugins import register_plugin_function

IntoExpr = Any

_INPUT_NAMES = ("scheme", "netloc", "path", "params", "query", "fragment")
_LIB = Path(__file__).parent
_NATIVE_EXTENSIONS = {".so", ".pyd", ".dll", ".dylib"}


def url_build(
    *,
    scheme: IntoExpr | None = None,
    netloc: IntoExpr | None = None,
    path: IntoExpr | None = None,
    params: IntoExpr | None = None,
    query: IntoExpr | None = None,
    fragment: IntoExpr | None = None,
) -> pl.Expr:
    """Build a URL from optional component expressions."""
    components = (scheme, netloc, path, params, query, fragment)
    parsed = [_optional_into_expr(component) for component in components]

    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="url_build",
            args=parsed,
            is_elementwise=True,
            use_abs_path=True,
        )

    row_expr = pl.struct(expr.alias(name) for expr, name in zip(parsed, _INPUT_NAMES, strict=True))
    return row_expr.map_elements(
        _url_build_row,
        return_dtype=pl.String,
        skip_nulls=False,
    )


def _url_build_row(row: Any) -> str:
    values = row if isinstance(row, Mapping) else {}
    components = tuple(values.get(name) or "" for name in _INPUT_NAMES)
    if not all(isinstance(component, str) for component in components):
        msg = "url_build expects String expressions"
        raise TypeError(msg)
    return urlunparse(components)


def _optional_into_expr(expr: IntoExpr | None) -> pl.Expr:
    if expr is None:
        return pl.lit("")
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, str):
        return pl.col(expr)
    return pl.lit(expr)


def _native_library_available() -> bool:
    return any(path.is_file() and path.suffix in _NATIVE_EXTENSIONS for path in _LIB.iterdir())

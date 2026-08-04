from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import polars as pl
from polars.plugins import register_plugin_function

IntoExpr = Any

_LIB = Path(__file__).parent
_NATIVE_EXTENSIONS = {".so", ".pyd", ".dll", ".dylib"}


def url_query_encode(expr: IntoExpr, *, doseq: bool = False) -> pl.Expr:
    """Encode a Struct or a list of key/value structs as a URL query string."""
    if not isinstance(doseq, bool):
        msg = "doseq must be a bool"
        raise TypeError(msg)

    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="url_query_encode",
            args=[expr],
            kwargs={"doseq": doseq},
            is_elementwise=True,
            use_abs_path=True,
        )

    parsed_expr = _parse_into_expr(expr)
    return parsed_expr.map_elements(
        lambda value: _url_query_encode_row(value, doseq=doseq),
        return_dtype=pl.String,
        skip_nulls=False,
    )


def _url_query_encode_row(value: Any, *, doseq: bool) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        query: Any = value
    elif isinstance(value, list):
        query = []
        for item in value:
            if not isinstance(item, Mapping) or "key" not in item or "value" not in item:
                msg = "url_query_encode expects list structs with `key` and `value` fields"
                raise TypeError(msg)
            query.append((item["key"], item["value"]))
    else:
        msg = "url_query_encode expects a Struct or List(Struct) expression"
        raise TypeError(msg)
    return urlencode(query, doseq=doseq)


def _parse_into_expr(expr: IntoExpr) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, str):
        return pl.col(expr)
    return pl.lit(expr)


def _native_library_available() -> bool:
    return any(path.is_file() and path.suffix in _NATIVE_EXTENSIONS for path in _LIB.iterdir())

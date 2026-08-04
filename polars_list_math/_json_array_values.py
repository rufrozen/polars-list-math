from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

IntoExpr = Any

_LIB = Path(__file__).parent
_NATIVE_EXTENSIONS = {".so", ".pyd", ".dll", ".dylib"}
_RETURN_DTYPE = pl.List(pl.String)


def json_array_values(expr: IntoExpr, *, strict: bool = False) -> pl.Expr:
    """Convert JSON arrays to lists of string values.

    String values are returned unchanged, null values stay null, and all other
    values are encoded as compact JSON. Valid non-array JSON values return null.
    Invalid JSON returns null unless ``strict`` is true.
    """
    if not isinstance(strict, bool):
        msg = "strict must be a bool"
        raise TypeError(msg)

    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="json_array_values",
            args=[expr],
            kwargs={"strict": strict},
            is_elementwise=True,
            use_abs_path=True,
        )

    parsed_expr = _parse_into_expr(expr)
    return parsed_expr.map_elements(
        lambda value: _json_array_values_row(value, strict=strict),
        return_dtype=_RETURN_DTYPE,
        skip_nulls=False,
    )


def install(*, overwrite: bool = False) -> None:
    """Register the Polars string namespace helper."""
    string_namespace = type(pl.col("__polars_list_math_json_array_values_probe__").str)

    if overwrite or not hasattr(string_namespace, "json_array_values"):
        string_namespace.json_array_values = _expr_str_json_array_values  # type: ignore


def _expr_str_json_array_values(self: Any, *, strict: bool = False) -> pl.Expr:
    base_expr = wrap_expr(self._pyexpr)
    return json_array_values(base_expr, strict=strict)


def _json_array_values_row(value: Any, *, strict: bool) -> list[str | None] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        msg = "json_array_values expects a String expression"
        raise TypeError(msg)

    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, UnicodeDecodeError):
        if strict:
            raise
        return None

    if not isinstance(decoded, list):
        return None

    return [item if isinstance(item, str) else _encode_json_value(item) for item in decoded]


def _encode_json_value(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse_into_expr(expr: IntoExpr) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, str):
        return pl.col(expr)
    return pl.lit(expr)


def _native_library_available() -> bool:
    return any(path.is_file() and path.suffix in _NATIVE_EXTENSIONS for path in _LIB.iterdir())

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

IntoExpr = Any

_INPUT_PREFIX = "__polars_list_math_list_combinations_input_"
_LIB = Path(__file__).parent
_NATIVE_EXTENSIONS = {".so", ".pyd", ".dll", ".dylib"}
_DEFAULT_LEFT_INDEX = "left_index"
_DEFAULT_RIGHT_INDEX = "right_index"


@dataclass(frozen=True)
class _CombinationFields:
    include_index: bool
    left_index: str
    right_index: str
    left_value: str
    right_value: str


def list_combinations(
    expr: IntoExpr,
    *,
    left_value: str = "left_value",
    right_value: str = "right_value",
    with_index: bool = False,
    left_index: str | None = None,
    right_index: str | None = None,
    skip_null: bool = False,
) -> pl.Expr:
    """Build a list expression with all ``i <= j`` pairs from each input list."""
    fields = _normalize_fields(left_index, right_index, left_value, right_value, with_index)
    _validate_skip_null(skip_null)
    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="list_combinations",
            args=[expr],
            kwargs=_kwargs(fields, with_index, skip_null),
            is_elementwise=True,
            use_abs_path=True,
        )

    parsed_expr = _parse_into_expr(expr)

    return parsed_expr.map_elements(
        lambda value: _combinations_row(value, fields, skip_null),
        return_dtype=None,
        skip_nulls=False,
    )


def list_combinations_to(
    expr: IntoExpr,
    target: IntoExpr,
    *,
    left_value: str = "left_value",
    right_value: str = "right_value",
    with_index: bool = False,
    left_index: str | None = None,
    right_index: str | None = None,
    skip_null: bool = False,
) -> pl.Expr:
    """Build a list expression with all pairs from the input list to a target list."""
    fields = _normalize_fields(left_index, right_index, left_value, right_value, with_index)
    _validate_skip_null(skip_null)
    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="list_combinations_to",
            args=[expr, target],
            kwargs=_kwargs(fields, with_index, skip_null),
            is_elementwise=True,
            use_abs_path=True,
        )

    parsed_exprs = [_parse_into_expr(expr), _parse_into_expr(target)]
    aliases = [f"{_INPUT_PREFIX}{index}" for index in range(2)]
    row_expr = pl.struct(
        parsed.alias(alias) for parsed, alias in zip(parsed_exprs, aliases, strict=True)
    )

    def combinations_to_row(row: Any) -> list[dict[str, Any]] | None:
        left, right = _row_values(row, aliases)
        return _combinations_to_row(left, right, fields, skip_null)

    return row_expr.map_elements(
        combinations_to_row,
        return_dtype=None,
        skip_nulls=False,
    )


def install(*, overwrite: bool = False) -> None:
    """Register the Polars list namespace helpers."""
    list_namespace = type(pl.col("__polars_list_math_list_combinations_probe__").list)

    if overwrite or not hasattr(list_namespace, "combinations"):
        list_namespace.combinations = _expr_list_combinations  # type: ignore
    if overwrite or not hasattr(list_namespace, "combinations_to"):
        list_namespace.combinations_to = _expr_list_combinations_to  # type: ignore


def _expr_list_combinations(
    self: Any,
    *,
    left_value: str = "left_value",
    right_value: str = "right_value",
    with_index: bool = False,
    left_index: str | None = None,
    right_index: str | None = None,
    skip_null: bool = False,
) -> pl.Expr:
    base_expr = wrap_expr(self._pyexpr)
    return list_combinations(
        base_expr,
        left_value=left_value,
        right_value=right_value,
        with_index=with_index,
        left_index=left_index,
        right_index=right_index,
        skip_null=skip_null,
    )


def _expr_list_combinations_to(
    self: Any,
    target: IntoExpr,
    *,
    left_value: str = "left_value",
    right_value: str = "right_value",
    with_index: bool = False,
    left_index: str | None = None,
    right_index: str | None = None,
    skip_null: bool = False,
) -> pl.Expr:
    base_expr = wrap_expr(self._pyexpr)
    return list_combinations_to(
        base_expr,
        target,
        left_value=left_value,
        right_value=right_value,
        with_index=with_index,
        left_index=left_index,
        right_index=right_index,
        skip_null=skip_null,
    )


def _combinations_row(
    value: Any,
    fields: _CombinationFields,
    skip_null: bool,
) -> list[dict[str, Any]] | None:
    values = _list_or_none(value, "list.combinations")
    if values is None:
        return None

    out: list[dict[str, Any]] = []
    for left_index in range(len(values)):
        for right_index in range(left_index, len(values)):
            left_value = values[left_index]
            right_value = values[right_index]
            if skip_null and (left_value is None or right_value is None):
                continue
            out.append(_pair_record(fields, left_index, right_index, left_value, right_value))
    return out


def _combinations_to_row(
    left: Any,
    right: Any,
    fields: _CombinationFields,
    skip_null: bool,
) -> list[dict[str, Any]] | None:
    left_values = _list_or_none(left, "list.combinations_to")
    right_values = _list_or_none(right, "list.combinations_to")
    if left_values is None or right_values is None:
        return None

    out: list[dict[str, Any]] = []
    for left_index, left_value in enumerate(left_values):
        for right_index, right_value in enumerate(right_values):
            if skip_null and (left_value is None or right_value is None):
                continue
            out.append(_pair_record(fields, left_index, right_index, left_value, right_value))
    return out


def _pair_record(
    fields: _CombinationFields,
    left_index: int,
    right_index: int,
    left_value: Any,
    right_value: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if fields.include_index:
        out[fields.left_index] = left_index
        out[fields.right_index] = right_index
    out[fields.left_value] = left_value
    out[fields.right_value] = right_value
    return out


def _normalize_fields(
    left_index: str | None,
    right_index: str | None,
    left_value: str,
    right_value: str,
    with_index: bool,
) -> _CombinationFields:
    _validate_with_index(with_index)
    include_index = with_index or left_index is not None or right_index is not None
    resolved_left_index = _DEFAULT_LEFT_INDEX if left_index is None else left_index
    resolved_right_index = _DEFAULT_RIGHT_INDEX if right_index is None else right_index

    index_names = (resolved_left_index, resolved_right_index) if include_index else ()
    field_names = (*index_names, left_value, right_value)
    if not all(isinstance(field, str) for field in field_names):
        msg = "combination field names must be strings"
        raise TypeError(msg)
    if not (left_index is None or isinstance(left_index, str)) or not (
        right_index is None or isinstance(right_index, str)
    ):
        msg = "combination index field names must be strings or None"
        raise TypeError(msg)
    if len(set(field_names)) != len(field_names):
        msg = "combination field names must be unique"
        raise ValueError(msg)
    return _CombinationFields(
        include_index=include_index,
        left_index=resolved_left_index,
        right_index=resolved_right_index,
        left_value=left_value,
        right_value=right_value,
    )


def _validate_with_index(with_index: bool) -> None:
    if not isinstance(with_index, bool):
        msg = "with_index must be a bool"
        raise TypeError(msg)


def _validate_skip_null(skip_null: bool) -> None:
    if not isinstance(skip_null, bool):
        msg = "skip_null must be a bool"
        raise TypeError(msg)


def _kwargs(fields: _CombinationFields, with_index: bool, skip_null: bool) -> dict[str, Any]:
    return {
        "left_index": fields.left_index if fields.include_index else None,
        "right_index": fields.right_index if fields.include_index else None,
        "left_value": fields.left_value,
        "right_value": fields.right_value,
        "with_index": with_index,
        "skip_null": skip_null,
    }


def _parse_into_expr(expr: IntoExpr) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, str):
        return pl.col(expr)
    return pl.lit(expr)


def _list_or_none(value: Any, name: str) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, pl.Series):
        return value.to_list()
    if not isinstance(value, list):
        msg = f"{name} expects a List expression"
        raise TypeError(msg)
    return value


def _row_values(row: Any, aliases: list[str]) -> list[Any]:
    if row is None:
        return [None] * len(aliases)
    if isinstance(row, Mapping):
        return [row[alias] for alias in aliases]
    return list(row)


def _native_library_available() -> bool:
    return any(path.is_file() and path.suffix in _NATIVE_EXTENSIONS for path in _LIB.iterdir())

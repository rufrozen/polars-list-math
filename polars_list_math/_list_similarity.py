from __future__ import annotations

import math
import struct
from collections.abc import Hashable, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from numbers import Real
from pathlib import Path
from typing import Any

import polars as pl
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

PolarsListInput = pl.Expr | str
DocumentId = bool | int | float | str | bytes | date | datetime | time | timedelta
DocumentValue = DocumentId | None
DocumentKey = tuple[type[Any], Hashable]

_INPUT_PREFIX = "__polars_list_math_list_similarity_input_"
_LIB = Path(__file__).parent
_NATIVE_EXTENSIONS = {".so", ".pyd", ".dll", ".dylib"}


def list_similarity(
    list_a: PolarsListInput,
    list_b: PolarsListInput,
    p: float = 0.9,
) -> pl.Expr:
    """Build a Polars expression with weighted Jaccard similarity for two lists."""
    _validate_polars_input(list_a, "list_a")
    _validate_polars_input(list_b, "list_b")
    p_value = _normalize_p(p)
    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="list_similarity",
            args=[list_a, list_b],
            kwargs={"p": p_value},
            is_elementwise=True,
            use_abs_path=True,
        )

    parsed_exprs = [_parse_into_expr(list_a), _parse_into_expr(list_b)]
    aliases = [f"{_INPUT_PREFIX}{index}" for index in range(2)]
    row_expr = pl.struct(
        expr.alias(alias) for expr, alias in zip(parsed_exprs, aliases, strict=True)
    )

    def score_row(row: Any) -> float | None:
        left, right = _row_values(row, aliases)
        if left is None or right is None:
            return None
        if not isinstance(left, list) or not isinstance(right, list):
            msg = "list_similarity expects expressions with List or Array dtype"
            raise TypeError(msg)
        return _weighted_jaccard_similarity(left, right, p_value)

    return row_expr.map_elements(
        score_row,
        return_dtype=pl.Float64,
        skip_nulls=False,
    )


def py_list_similarity(
    list_a: Sequence[DocumentValue],
    list_b: Sequence[DocumentValue],
    p: float = 0.9,
) -> float:
    """Compute weighted Jaccard similarity for two Python sequences."""
    return _weighted_jaccard_similarity(list_a, list_b, _normalize_p(p))


def install(*, overwrite: bool = False) -> None:
    """Register the Polars list namespace helper."""
    list_namespace = type(pl.col("__polars_list_math_list_similarity_probe__").list)

    if overwrite or not hasattr(list_namespace, "similarity"):
        list_namespace.similarity = _expr_list_similarity  # type: ignore


def _expr_list_similarity(
    self: Any,
    other: PolarsListInput,
    p: float = 0.9,
) -> pl.Expr:
    base_expr = wrap_expr(self._pyexpr)
    return list_similarity(base_expr, other, p=p)


def _weighted_jaccard_similarity(
    list_a: Sequence[DocumentValue],
    list_b: Sequence[DocumentValue],
    p: float,
) -> float:
    weights_a, dtype_a = _document_weights(list_a, p)
    weights_b, dtype_b = _document_weights(list_b, p)
    _ensure_compatible_document_types(dtype_a, dtype_b)

    numerator = 0.0
    denominator = 0.0

    remaining_b = weights_b.copy()
    for document_id, weight_a in weights_a.items():
        weight_b = remaining_b.pop(document_id, 0.0)
        numerator += min(weight_a, weight_b)
        denominator += max(weight_a, weight_b)

    denominator += sum(remaining_b.values())
    if denominator == 0.0:
        return 1.0
    return numerator / denominator


def _document_weights(
    values: Sequence[DocumentValue],
    p: float,
) -> tuple[dict[DocumentKey, float], type[Any] | None]:
    weights: dict[DocumentKey, float] = {}
    document_type: type[Any] | None = None

    for index, document_id in enumerate(values):
        if document_id is None:
            continue

        current_type = _document_type(document_id)
        if document_type is None:
            document_type = current_type
        elif document_type is not current_type:
            msg = "document_id values must not mix primitive scalar types"
            raise TypeError(msg)

        document_key = _document_key(document_id)
        if document_key not in weights:
            weights[document_key] = p**index

    return weights, document_type


def _document_key(document_id: Any) -> DocumentKey:
    document_type = _document_type(document_id)
    if document_type is float:
        return document_type, struct.pack("!d", document_id)
    return document_type, document_id


def _document_type(document_id: Any) -> type[Any]:
    if not isinstance(document_id, (bool, int, float, str, bytes, date, datetime, time, timedelta)):
        msg = "document_id values must be primitive scalar values or None"
        raise TypeError(msg)
    return type(document_id)


def _ensure_compatible_document_types(
    dtype_a: type[Any] | None,
    dtype_b: type[Any] | None,
) -> None:
    if dtype_a is not None and dtype_b is not None and dtype_a is not dtype_b:
        msg = "document_id values must not mix primitive scalar types"
        raise TypeError(msg)


def _normalize_p(p: float) -> float:
    if isinstance(p, bool) or not isinstance(p, Real):
        msg = "p must be a real number"
        raise TypeError(msg)

    p_value = float(p)
    if not math.isfinite(p_value) or not 0.0 < p_value <= 1.0:
        msg = "p must satisfy 0 < p <= 1"
        raise ValueError(msg)
    return p_value


def _validate_polars_input(value: object, name: str) -> None:
    if not isinstance(value, (pl.Expr, str)):
        msg = (
            f"{name} must be a Polars expression or column name; "
            "use py_list_similarity for Python sequences"
        )
        raise TypeError(msg)


def _parse_into_expr(expr: PolarsListInput) -> pl.Expr:
    if isinstance(expr, pl.Expr):
        return expr
    return pl.col(expr)


def _row_values(row: Any, aliases: Sequence[str]) -> list[Any]:
    if row is None:
        return [None] * len(aliases)
    if isinstance(row, Mapping):
        return [row[alias] for alias in aliases]
    return list(row)


def _native_library_available() -> bool:
    return any(path.is_file() and path.suffix in _NATIVE_EXTENSIONS for path in _LIB.iterdir())

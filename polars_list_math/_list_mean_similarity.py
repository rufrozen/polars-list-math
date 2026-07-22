from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

from ._list_similarity import (
    _native_library_available,
    _normalize_p,
    _parse_into_expr,
    _row_values,
    _validate_polars_input,
    _weighted_jaccard_similarity,
)

PolarsNestedListInput = pl.Expr | str

_INPUT_PREFIX = "__polars_list_math_list_mean_similarity_input_"
_LIB = Path(__file__).parent


def list_mean_similarity(
    lists: PolarsNestedListInput,
    p: float = 0.9,
) -> pl.Expr:
    """Build a Polars expression with mean similarity for each list in a list."""
    _validate_polars_input(lists, "lists")
    p_value = _normalize_p(p)
    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="list_mean_similarity",
            args=[lists],
            kwargs={"p": p_value},
            is_elementwise=True,
            use_abs_path=True,
        )

    return _parse_into_expr(lists).map_elements(
        lambda value: _mean_similarity_self(value, p_value),
        return_dtype=pl.List(pl.Float64),
        skip_nulls=False,
    )


def list_mean_similarity_to(
    lists: PolarsNestedListInput,
    reference_lists: PolarsNestedListInput,
    p: float = 0.9,
) -> pl.Expr:
    """Build a Polars expression with mean similarity to reference lists."""
    _validate_polars_input(lists, "lists")
    _validate_polars_input(reference_lists, "reference_lists")
    p_value = _normalize_p(p)
    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="list_mean_similarity_to",
            args=[lists, reference_lists],
            kwargs={"p": p_value},
            is_elementwise=True,
            use_abs_path=True,
        )

    parsed_exprs = [_parse_into_expr(lists), _parse_into_expr(reference_lists)]
    aliases = [f"{_INPUT_PREFIX}{index}" for index in range(2)]
    row_expr = pl.struct(
        expr.alias(alias) for expr, alias in zip(parsed_exprs, aliases, strict=True)
    )

    def score_row(row: Any) -> list[float | None] | None:
        target, reference = _row_values(row, aliases)
        return _mean_similarity_to(target, reference, p_value)

    return row_expr.map_elements(
        score_row,
        return_dtype=pl.List(pl.Float64),
        skip_nulls=False,
    )


def install(*, overwrite: bool = False) -> None:
    """Register the Polars list namespace helpers."""
    list_namespace = type(pl.col("__polars_list_math_list_mean_similarity_probe__").list)

    if overwrite or not hasattr(list_namespace, "mean_similarity"):
        list_namespace.mean_similarity = _expr_list_mean_similarity  # type: ignore
    if overwrite or not hasattr(list_namespace, "mean_similarity_to"):
        list_namespace.mean_similarity_to = _expr_list_mean_similarity_to  # type: ignore


def _expr_list_mean_similarity(
    self: Any,
    p: float = 0.9,
) -> pl.Expr:
    base_expr = wrap_expr(self._pyexpr)
    return list_mean_similarity(base_expr, p=p)


def _expr_list_mean_similarity_to(
    self: Any,
    reference_lists: PolarsNestedListInput,
    p: float = 0.9,
) -> pl.Expr:
    base_expr = wrap_expr(self._pyexpr)
    return list_mean_similarity_to(base_expr, reference_lists, p=p)


def _mean_similarity_self(value: Any, p: float) -> list[float | None] | None:
    if value is None:
        return None
    lists = _validate_lists(value, "lists")

    weights = [_list_or_none(item, "lists") for item in lists]
    sums = [0.0] * len(weights)
    counts = [0] * len(weights)

    for left_index in range(len(weights)):
        left = weights[left_index]
        if left is None:
            continue
        for right_index in range(left_index + 1, len(weights)):
            right = weights[right_index]
            if right is None:
                continue

            score = _weighted_jaccard_similarity(left, right, p)
            sums[left_index] += score
            sums[right_index] += score
            counts[left_index] += 1
            counts[right_index] += 1

    return [
        sums[index] / counts[index] if weights[index] is not None and counts[index] else None
        for index in range(len(weights))
    ]


def _mean_similarity_to(
    target_value: Any,
    reference_value: Any,
    p: float,
) -> list[float | None] | None:
    if target_value is None:
        return None
    target_lists = _validate_lists(target_value, "lists")

    target_lists = [_list_or_none(item, "lists") for item in target_lists]
    if reference_value is None:
        return _null_scores(len(target_lists))

    reference_lists = [
        item
        for item in (
            _list_or_none(item, "reference_lists")
            for item in _validate_lists(reference_value, "reference_lists")
        )
        if item is not None
    ]
    if not reference_lists:
        return _null_scores(len(target_lists))

    out: list[float | None] = []
    for target_list in target_lists:
        if target_list is None:
            out.append(None)
            continue

        total = sum(
            _weighted_jaccard_similarity(target_list, reference_list, p)
            for reference_list in reference_lists
        )
        out.append(total / len(reference_lists))

    return out


def _validate_lists(value: Any, name: str) -> list[Any]:
    if isinstance(value, pl.Series):
        return value.to_list()
    if not isinstance(value, list):
        msg = f"{name} must be a List(List(primitive)) expression"
        raise TypeError(msg)
    return value


def _null_scores(length: int) -> list[float | None]:
    return [None for _ in range(length)]


def _list_or_none(value: Any, name: str) -> Sequence[Any] | None:
    if value is None:
        return None
    if isinstance(value, pl.Series):
        return value.to_list()
    if not isinstance(value, list):
        msg = f"{name} must contain inner lists or null values"
        raise TypeError(msg)
    return value

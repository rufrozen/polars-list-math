from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

from ._list_mean_similarity import _list_or_none, _validate_lists
from ._list_similarity import (
    DocumentKey,
    DocumentValue,
    PolarsListInput,
    _document_weights,
    _ensure_compatible_document_types,
    _native_library_available,
    _parse_into_expr,
    _row_values,
)

_LIB = Path(__file__).parent
_INPUT_NAMES = ("__containment_left", "__containment_right")
DocumentSet = tuple[set[DocumentKey], type[Any] | None]


def py_list_containment(list_a: Sequence[DocumentValue], list_b: Sequence[DocumentValue]) -> float:
    """Return the fraction of unique non-null IDs in list_a also present in list_b."""
    return _score(_document_set(list_a), _document_set(list_b))


def list_containment(list_a: PolarsListInput, list_b: PolarsListInput) -> pl.Expr:
    """Compare lists row by row, returning Float64; an empty left list scores zero."""
    _validate_input(list_a, "list_a")
    _validate_input(list_b, "list_b")
    if _native_library_available():
        return _plugin("list_containment", [list_a, list_b])

    def score_row(row: Any) -> float | None:
        left, right = _row_values(row, _INPUT_NAMES)
        if left is None or right is None:
            return None
        return py_list_containment(
            _validate_lists(left, "list_a"), _validate_lists(right, "list_b")
        )

    return _pair(list_a, list_b).map_elements(score_row, return_dtype=pl.Float64, skip_nulls=False)


def list_mean_containment(lists: PolarsListInput) -> pl.Expr:
    """Return each inner list's mean outgoing containment against its peers."""
    _validate_input(lists, "lists")
    if _native_library_available():
        return _plugin("list_mean_containment", [lists])
    return _parse_into_expr(lists).map_elements(
        _mean_self, return_dtype=pl.List(pl.Float64), skip_nulls=False
    )


def list_mean_containment_to(lists: PolarsListInput, reference_lists: PolarsListInput) -> pl.Expr:
    """Return each target list's mean containment against the reference lists."""
    _validate_input(lists, "lists")
    _validate_input(reference_lists, "reference_lists")
    if _native_library_available():
        return _plugin("list_mean_containment_to", [lists, reference_lists])

    def score_row(row: Any) -> list[float | None] | None:
        targets, references = _row_values(row, _INPUT_NAMES)
        return _mean_to(targets, references)

    return _pair(lists, reference_lists).map_elements(
        score_row, return_dtype=pl.List(pl.Float64), skip_nulls=False
    )


def _document_set(values: Sequence[DocumentValue]) -> DocumentSet:
    weights, dtype = _document_weights(values, 1.0)
    return set(weights), dtype


def _score(left: DocumentSet, right: DocumentSet) -> float:
    _ensure_compatible_document_types(left[1], right[1])
    return len(left[0] & right[0]) / len(left[0]) if left[0] else 0.0


def _sets(value: Any, name: str) -> list[DocumentSet | None]:
    return [
        _document_set(items) if (items := _list_or_none(item, name)) is not None else None
        for item in _validate_lists(value, name)
    ]


def _mean_self(value: Any) -> list[float | None] | None:
    if value is None:
        return None
    sets = _sets(value, "lists")
    out: list[float | None] = []
    for index, target in enumerate(sets):
        peers = [other for j, other in enumerate(sets) if j != index and other is not None]
        out.append(
            sum(_score(target, other) for other in peers) / len(peers)
            if target is not None and peers
            else None
        )
    return out


def _mean_to(targets: Any, references: Any) -> list[float | None] | None:
    if targets is None:
        return None
    target_sets = _sets(targets, "lists")
    reference_sets = (
        [item for item in _sets(references, "reference_lists") if item is not None]
        if references is not None
        else []
    )
    return [
        sum(_score(target, other) for other in reference_sets) / len(reference_sets)
        if target is not None and reference_sets
        else None
        for target in target_sets
    ]


def _validate_input(value: object, name: str) -> None:
    if not isinstance(value, (pl.Expr, str)):
        msg = (
            f"{name} must be a Polars expression or column name; "
            "use py_list_containment for Python sequences"
        )
        raise TypeError(msg)


def _pair(left: PolarsListInput, right: PolarsListInput) -> pl.Expr:
    return pl.struct(
        _parse_into_expr(left).alias(_INPUT_NAMES[0]),
        _parse_into_expr(right).alias(_INPUT_NAMES[1]),
    )


def _plugin(name: str, args: list[PolarsListInput]) -> pl.Expr:
    return register_plugin_function(
        plugin_path=_LIB, function_name=name, args=args, is_elementwise=True, use_abs_path=True
    )


def install(*, overwrite: bool = False) -> None:
    """Register the Polars list namespace containment helpers."""
    namespace = type(pl.col("__containment_probe").list)
    for name, method in (
        ("containment", _expr_containment),
        ("mean_containment", _expr_mean_containment),
        ("mean_containment_to", _expr_mean_containment_to),
    ):
        if overwrite or not hasattr(namespace, name):
            setattr(namespace, name, method)


def _expr_containment(self: Any, other: PolarsListInput) -> pl.Expr:
    return list_containment(wrap_expr(self._pyexpr), other)


def _expr_mean_containment(self: Any) -> pl.Expr:
    return list_mean_containment(wrap_expr(self._pyexpr))


def _expr_mean_containment_to(self: Any, reference_lists: PolarsListInput) -> pl.Expr:
    return list_mean_containment_to(wrap_expr(self._pyexpr), reference_lists)

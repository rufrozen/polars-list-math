from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
from polars._utils.wrap import wrap_expr
from polars.plugins import register_plugin_function

from ._list_similarity import (
    DocumentKey,
    DocumentValue,
    PolarsListInput,
    _document_weights,
    _ensure_compatible_document_types,
    _native_library_available,
    _parse_into_expr,
)

_LIB = Path(__file__).parent
Ranks = tuple[dict[DocumentKey, int], type[Any] | None]


def list_kendall_tau(
    list_a: PolarsListInput,
    list_b: PolarsListInput,
    *,
    normalize: bool = False,
) -> pl.Expr:
    """Compare the order of shared IDs; fewer than two shared IDs returns null.

    Duplicates use their first occurrence and null items are ignored. Set
    normalize=True to map the coefficient from [-1, 1] to [0, 1].
    """
    return _expression("list_kendall_tau", [list_a, list_b], normalize, depth=1)


def list_mean_kendall_tau(
    lists: PolarsListInput,
    *,
    normalize: bool = False,
) -> pl.Expr:
    """Return each inner list's mean tau against its peers, excluding undefined pairs."""
    return _expression("list_mean_kendall_tau", [lists], normalize, depth=2)


def list_mean_kendall_tau_to(
    lists: PolarsListInput,
    reference_lists: PolarsListInput,
    *,
    normalize: bool = False,
) -> pl.Expr:
    """Return each inner list's mean tau against references, excluding undefined pairs."""
    return _expression("list_mean_kendall_tau_to", [lists, reference_lists], normalize, depth=2)


def py_list_kendall_tau(
    list_a: Sequence[DocumentValue],
    list_b: Sequence[DocumentValue],
    *,
    normalize: bool = False,
) -> float | None:
    """Compute Kendall tau of the common IDs' positions in two Python sequences.

    Return None when fewer than two distinct non-null IDs are shared. ID types
    and first-occurrence handling match py_list_similarity.
    """
    _validate_normalize(normalize)
    return _score(_ranks(list_a), _ranks(list_b), normalize)


def install(*, overwrite: bool = False) -> None:
    """Register the Polars list namespace helpers."""
    namespace = type(pl.col("__polars_list_math_kendall_tau_probe__").list)
    methods = {
        "kendall_tau": _expr_list_kendall_tau,
        "mean_kendall_tau": _expr_list_mean_kendall_tau,
        "mean_kendall_tau_to": _expr_list_mean_kendall_tau_to,
    }
    for name, method in methods.items():
        if overwrite or not hasattr(namespace, name):
            setattr(namespace, name, method)


def _expr_list_kendall_tau(
    self: Any, other: PolarsListInput, *, normalize: bool = False
) -> pl.Expr:
    return list_kendall_tau(wrap_expr(self._pyexpr), other, normalize=normalize)


def _expr_list_mean_kendall_tau(self: Any, *, normalize: bool = False) -> pl.Expr:
    return list_mean_kendall_tau(wrap_expr(self._pyexpr), normalize=normalize)


def _expr_list_mean_kendall_tau_to(
    self: Any, reference_lists: PolarsListInput, *, normalize: bool = False
) -> pl.Expr:
    return list_mean_kendall_tau_to(wrap_expr(self._pyexpr), reference_lists, normalize=normalize)


def _validate_normalize(normalize: bool) -> None:
    if not isinstance(normalize, bool):
        msg = "normalize must be a bool"
        raise TypeError(msg)


def _expression(
    name: str, inputs: list[PolarsListInput], normalize: bool, *, depth: int
) -> pl.Expr:
    _validate_normalize(normalize)
    for value in inputs:
        if not isinstance(value, (pl.Expr, str)):
            msg = (
                "inputs must be Polars expressions or column names; "
                "use py_list_kendall_tau for Python sequences"
            )
            raise TypeError(msg)
    if _native_library_available():
        return register_plugin_function(
            plugin_path=_LIB,
            function_name=name,
            args=inputs,
            kwargs={"normalize": normalize},
            is_elementwise=True,
            use_abs_path=True,
        )

    output_dtype = pl.Float64 if depth == 1 else pl.List(pl.Float64)

    def score_batch(columns: Sequence[pl.Series]) -> pl.Series:
        _validate_dtypes(columns, depth)
        rows = next((len(column) for column in columns if len(column) != 1), 1)
        if any(len(column) not in (1, rows) for column in columns):
            msg = "input column lengths must match or be 1"
            raise pl.exceptions.ShapeError(msg)
        # Keep nanosecond temporal IDs distinct instead of truncating them to
        # Python datetime/timedelta's microsecond precision.
        values = [column.to_physical().to_list() for column in columns]
        scores = []
        for index in range(rows):
            row = [column[0 if len(column) == 1 else index] for column in values]
            if depth == 1:
                left, right = row
                scores.append(
                    None
                    if left is None or right is None
                    else _score(_ranks(left), _ranks(right), normalize)
                )
            else:
                scores.append(
                    _mean_scores(
                        row[0],
                        row[1] if len(row) == 2 else None,
                        normalize,
                        self_compare=len(row) == 1,
                    )
                )
        return pl.Series(scores, dtype=output_dtype)

    return pl.map_batches(
        [_parse_into_expr(value) for value in inputs],
        score_batch,
        return_dtype=output_dtype,
        is_elementwise=True,
    )


def _validate_dtypes(columns: Sequence[pl.Series], depth: int) -> None:
    leaf_dtypes = []
    for column in columns:
        dtype = column.dtype
        for _ in range(depth):
            if not isinstance(dtype, (pl.List, pl.Array)):
                msg = f"Kendall tau expects {depth} List or Array levels"
                raise TypeError(msg)
            dtype = dtype.inner
        if not (
            dtype in (pl.Null, pl.Boolean, pl.String, pl.Binary)
            or dtype.is_integer()
            or dtype.is_float()
            or dtype.is_temporal()
        ):
            msg = "Kendall tau expects a primitive list inner dtype"
            raise TypeError(msg)
        if dtype != pl.Null:
            leaf_dtypes.append(dtype)
    if leaf_dtypes and any(dtype != leaf_dtypes[0] for dtype in leaf_dtypes):
        msg = "document_id list inner dtypes must match exactly or be Null"
        raise TypeError(msg)


def _ranks(values: Sequence[DocumentValue]) -> Ranks:
    weights, dtype = _document_weights(values, 1.0)
    return {key: index for index, key in enumerate(weights)}, dtype


def _score(left: Ranks, right: Ranks, normalize: bool) -> float | None:
    ranks_a, dtype_a = left
    ranks_b, dtype_b = right
    _ensure_compatible_document_types(dtype_a, dtype_b)
    # Dictionaries preserve first-occurrence order. Count inversions of the
    # matching right-hand positions with a Fenwick tree.
    positions = [ranks_b[key] for key in ranks_a if key in ranks_b]
    count = len(positions)
    if count < 2:
        return None
    tree = [0] * (len(ranks_b) + 1)
    discordant = 0
    for seen, position in enumerate(positions):
        index = position + 1
        prefix = 0
        while index:
            prefix += tree[index]
            index -= index & -index
        discordant += seen - prefix
        index = position + 1
        while index < len(tree):
            tree[index] += 1
            index += index & -index
    pairs = count * (count - 1) // 2
    tau = 1.0 - 2.0 * discordant / pairs
    return (tau + 1.0) / 2.0 if normalize else tau


def _mean_scores(
    target: list[Any] | None,
    reference: list[Any] | None,
    normalize: bool,
    *,
    self_compare: bool,
) -> list[float | None] | None:
    if target is None:
        return None
    targets = [None if item is None else _ranks(item) for item in target]
    references = (
        targets
        if self_compare
        else [_ranks(item) for item in (reference or []) if item is not None]
    )
    sums = [0.0] * len(targets)
    counts = [0] * len(targets)
    for i, left in enumerate(targets):
        if left is None:
            continue
        for j in range(i + 1 if self_compare else 0, len(references)):
            right = references[j]
            if right is None:
                continue
            score = _score(left, right, normalize)
            if score is None:
                continue
            sums[i] += score
            counts[i] += 1
            if self_compare:
                sums[j] += score
                counts[j] += 1
    return [total / count if count else None for total, count in zip(sums, counts, strict=True)]

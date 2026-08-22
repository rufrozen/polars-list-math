from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from types import ModuleType
from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal

ROW_COUNT = int(os.environ.get("POLARS_LIST_MATH_PERF_ROWS", "2000"))
REPEAT_COUNT = int(os.environ.get("POLARS_LIST_MATH_PERF_REPEATS", "3"))


@dataclass(frozen=True)
class PerformanceCase:
    name: str
    module: ModuleType
    frame: pl.DataFrame
    expression: Callable[[], pl.Expr]


def plugin_module(name: str) -> ModuleType:
    return import_module(f"polars_list_math.{name}")


def run_case(case: PerformanceCase) -> None:
    if not case.module._native_library_available():
        pytest.skip("native extension is not built; run `make develop`")

    native_elapsed, native_result = _measure(_plan(case, native=True))
    fallback_elapsed, fallback_result = _measure(_plan(case, native=False))

    # Combination indices are UInt32 in Rust and inferred as Int64 by Python.
    assert_frame_equal(native_result, fallback_result, check_dtypes=False)
    speedup = fallback_elapsed / native_elapsed
    print(
        f"{case.name:>25} rows={ROW_COUNT:>7}: native={native_elapsed:.6f}s "
        f"fallback={fallback_elapsed:.6f}s speedup={speedup:.2f}x"
    )


def _plan(case: PerformanceCase, *, native: bool) -> pl.LazyFrame:
    with patch.object(case.module, "_native_library_available", return_value=native):
        expression = case.expression()
    return case.frame.lazy().select(expression.alias("result"))


def _measure(plan: pl.LazyFrame) -> tuple[float, pl.DataFrame]:
    result = plan.collect()
    durations: list[float] = []
    for _ in range(REPEAT_COUNT):
        started_at = perf_counter()
        result = plan.collect()
        durations.append(perf_counter() - started_at)
    return min(durations), result

"""Reproducible full-operation benchmark and explode/implode reference."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from polars_list_math import _nested_join, nested_join


@dataclass
class Case:
    left: pl.DataFrame
    right: pl.DataFrame
    on: dict[str, str]
    shape: str
    values: dict[str, str]
    indexes: list[str]
    leaves: int
    skip_null_keys: bool = False


def add_indexes(frame: pl.DataFrame, shape: str) -> tuple[pl.DataFrame, list[str]]:
    """Generate positional indexes before any operation is timed."""
    depth = _nested_join._depth(frame.schema[shape])[0]

    def positions(node, remaining):
        if node is None:
            return None
        if remaining == 1:
            return list(range(len(node)))
        return [positions(child, remaining - 1) for child in node]

    columns = [pl.Series("__row", range(frame.height), dtype=pl.UInt32)]
    data = frame[shape].to_list()
    for level in range(1, depth + 1):
        columns.append(
            pl.Series(
                f"__pos_{level}",
                [positions(node, level) for node in data],
                dtype=_nested_join._wrap(pl.UInt32, level),
            )
        )
    return frame.with_columns(columns), [s.name for s in columns]


def explode_join_implode(case: Case) -> pl.DataFrame:
    """Reference using existing indexes, including null/empty-list reconstruction."""
    left, right, on, shape, values, indexes = (
        case.left,
        case.right,
        case.on,
        case.shape,
        case.values,
        case.indexes,
    )
    on, values = _nested_join._arguments(on, shape, values, case.skip_null_keys)
    if set(values.values()).intersection(left.columns):
        raise pl.exceptions.DuplicateError("output names already exist in left")
    _nested_join._output_dtype(
        [
            pl.List(pl.Struct(dict(right.schema))),
            left.schema[shape],
            *(left.schema[name] for name in on),
        ],
        on,
        values,
    )
    right_keys = list(dict.fromkeys(on.values()))
    null_right = pl.any_horizontal(pl.col(name).is_null() for name in right_keys)
    if case.skip_null_keys:
        right = right.filter(~null_right)
    elif right.select(null_right.any()).item():
        raise pl.exceptions.ComputeError("null in right key")
    if right.select(right_keys).is_duplicated().any():
        raise pl.exceptions.ComputeError("duplicate right key")

    depth = len(indexes) - 1
    key_depths = {name: _nested_join._depth(left.schema[name])[0] for name in on}
    index_depths = {name: level for level, name in enumerate(indexes)}
    selected = list(dict.fromkeys([*indexes, *on]))
    plan = (
        left.lazy().select(selected).with_columns(pl.col(shape).alias("__shape"))
        if shape in selected
        else left.lazy().select(*selected, pl.col(shape).alias("__shape"))
    )
    plan = plan.with_columns(pl.lit(False).alias("__bad_structure"))
    for level in range(1, depth + 1):
        descendants = [name for name, d in key_depths.items() if d >= level]
        checks = [
            pl.col("__shape").is_not_null()
            & (pl.col(name).is_null() | (pl.col(name).list.len() != pl.col("__shape").list.len()))
            for name in descendants
        ]
        plan = plan.with_columns(
            pl.col("__shape").is_null().alias(f"__null_{level}"),
            pl.col("__shape").list.len().alias(f"__len_{level}"),
            pl.any_horizontal(pl.col("__bad_structure"), *checks).alias("__bad_structure"),
        )
        # A null shape branch has no children; deeper key data is irrelevant.
        # Mask it before multi-column explode to avoid inventing extra positions.
        plan = plan.with_columns(
            pl.when(pl.col("__shape").is_null()).then(None).otherwise(pl.col(name)).alias(name)
            for name in descendants
        )
        columns = list(
            dict.fromkeys(
                ["__shape", *descendants, *(name for name, d in index_depths.items() if d >= level)]
            )
        )
        plan = plan.explode(columns, empty_as_null=True)
    flat = plan.collect()
    if flat["__bad_structure"].any():
        raise pl.exceptions.ShapeError("key does not align with shape")
    active = pl.all_horizontal(pl.col(name).is_not_null() for name in indexes)
    null_left = pl.any_horizontal(pl.col(name).is_null() for name in on)
    if not case.skip_null_keys and flat.select((active & null_left).any()).item():
        raise pl.exceptions.ComputeError("null in left key")

    right_aliases = [f"__right_key_{i}" for i in range(len(on))]
    renamed_right = right.select(
        *(
            pl.col(name).alias(alias)
            for name, alias in zip(on.values(), right_aliases, strict=True)
        ),
        *(pl.col(name).alias(alias) for name, alias in values.items()),
    )
    merged = flat.lazy().join(
        renamed_right.lazy(),
        left_on=list(on),
        right_on=right_aliases,
        how="left",
        maintain_order="left",
    )
    # Leaf placeholders from empty/null lists must never acquire a value.
    merged = merged.with_columns(
        pl.when(active).then(pl.col(alias)).otherwise(None).alias(alias)
        for alias in values.values()
    )
    for level in range(depth, 0, -1):
        flags = [
            name
            for parent in range(1, level + 1)
            for name in (f"__null_{parent}", f"__len_{parent}")
        ]
        merged = (
            merged.group_by(indexes[:level], maintain_order=True)
            .agg(
                *(pl.col(alias) for alias in values.values()),
                *(pl.col(name).first() for name in flags),
            )
            .with_columns(
                pl.when(pl.col(f"__null_{level}"))
                .then(pl.lit(None, dtype=_nested_join._wrap(right.schema[name], depth - level + 1)))
                .when(pl.col(f"__len_{level}") == 0)
                .then(pl.lit([], dtype=_nested_join._wrap(right.schema[name], depth - level + 1)))
                .otherwise(pl.col(alias))
                .alias(alias)
                for name, alias in values.items()
            )
        )
    output = merged.select(indexes[0], *values.values()).collect()
    return left.join(output, on=indexes[0], how="left", maintain_order="left", validate="1:1")


def make_case(rows: int, scenario: str) -> Case:
    """Build deterministic ragged SERPs or four-level dialogs, before timing."""
    depth = 2 if scenario == "serp" else 4
    sizes = [3, 12] if depth == 2 else [2, 2, 2, 8]
    shapes = []
    records: dict[tuple, dict] = {}
    leaves = 0
    contexts = max(1, rows // 4)

    def shape_node(row, level, path):
        code = row * 17 + sum((i + 3) * (v + 1) for i, v in enumerate(path))
        if level == depth:
            # Repeated IDs within a list and about 1/9 missing matches.
            return -(path[-1] + 1) if code % 9 == 0 else path[-1] // 2 + 1
        if code % 31 == 0:
            return None
        if code % 19 == 0:
            return []
        count = sizes[level] - code % 2
        if scenario == "skew" and level == depth - 1 and code % 7 == 0:
            count *= 10
        return [shape_node(row, level + 1, [*path, i]) for i in range(count)]

    def at_level(node, target, function, path=()):
        if len(path) == target:
            return function(path)
        if node is None:
            return None
        return [at_level(child, target, function, (*path, i)) for i, child in enumerate(node)]

    def record_leaves(node, row, path=()):
        nonlocal leaves
        if len(path) == depth:
            leaves += 1
            if node < 0:
                return
            if depth == 2:
                key = (f"q{row % contexts}", node)
                record = dict(zip(["query", "id"], key, strict=True))
            else:
                key = (f"c{row % contexts}:{path[0]}", f"m{path[1]}", path[2], node)
                record = dict(zip(["conv", "msg", "block", "id"], key, strict=True))
            record.update(label=[3, 5, 7, 8][(node + row % contexts) % 4], score=node / 10)
            records[key] = record
            return
        if node is not None:
            for i, child in enumerate(node):
                record_leaves(child, row, (*path, i))

    for row in range(rows):
        node = shape_node(row, 0, [])
        shapes.append(node)
        record_leaves(node, row)
    data: dict[str, Any] = {"sku": pl.Series(shapes, dtype=_nested_join._wrap(pl.Int64, depth))}
    if depth == 2:
        data["query"] = pl.Series([f"q{r % contexts}" for r in range(rows)], dtype=pl.String)
        on = {"query": "query", "sku": "id"}
        right_schema = {"query": pl.String, "id": pl.Int64, "label": pl.Int8, "score": pl.Float64}
    else:
        for name, target, dtype in [
            ("conv", 1, pl.String),
            ("msg", 2, pl.String),
            ("block", 3, pl.UInt32),
        ]:
            contents = []
            for row, node in enumerate(shapes):

                def key(path, name=name, row=row):
                    if name == "conv":
                        return f"c{row % contexts}:{path[0]}"
                    if name == "msg":
                        return f"m{path[1]}"
                    return path[2]

                contents.append(at_level(node, target, key))
            data[name] = pl.Series(contents, dtype=_nested_join._wrap(dtype, target))
        on = {"conv": "conv", "msg": "msg", "block": "block", "sku": "id"}
        right_schema = {
            "conv": pl.String,
            "msg": pl.String,
            "block": pl.UInt32,
            "id": pl.Int64,
            "label": pl.Int8,
            "score": pl.Float64,
        }
    left, indexes = add_indexes(pl.DataFrame(data), "sku")
    right = pl.DataFrame(list(records.values()), schema=right_schema)
    return Case(left, right, on, "sku", {"label": "label", "score": "score"}, indexes, leaves)


def run_strategy(case: Case, strategy: str) -> pl.DataFrame:
    if strategy == "explode":
        return explode_join_implode(case)
    with patch.object(_nested_join, "_native_library_available", return_value=strategy == "native"):
        return nested_join(
            case.left,
            case.right,
            on=case.on,
            shape=case.shape,
            values=case.values,
            skip_null_keys=case.skip_null_keys,
        )


def _memory_mib() -> tuple[float | None, float | None]:
    # VmHWM is reset on exec. getrusage(RUSAGE_SELF).ru_maxrss can inherit the
    # parent's pre-exec high-water mark and would invalidate worker comparisons.
    status = Path("/proc/self/status")
    if not status.exists():
        return None, None
    metrics = {
        line.split(":", 1)[0]: float(line.split()[1]) / 1024
        for line in status.read_text().splitlines()
        if line.startswith(("VmRSS:", "VmHWM:"))
    }
    return metrics["VmRSS"], metrics["VmHWM"]


def worker(rows: int, scenario: str, strategy: str, repeats: int) -> dict:
    case = make_case(rows, scenario)
    gc.collect()
    baseline_rss, _ = _memory_mib()
    durations = []
    for iteration in range(repeats + 1):
        start = perf_counter()
        result = run_strategy(case, strategy)
        elapsed = perf_counter() - start
        if iteration:
            durations.append(elapsed)
        del result
        gc.collect()
    _, peak_rss = _memory_mib()
    best = min(durations)
    return {
        "scenario": scenario,
        "strategy": strategy,
        "rows": rows,
        "right_rows": case.right.height,
        "leaves": case.leaves,
        "seconds": best,
        "median_seconds": statistics.median(durations),
        "leaves_per_second": case.leaves / best,
        "baseline_rss_mib": baseline_rss,
        "peak_rss_mib": peak_rss,
        "peak_growth_mib": peak_rss - baseline_rss
        if peak_rss is not None and baseline_rss is not None
        else None,
    }


def benchmark(rows: int, scenario: str, repeats: int) -> list[dict]:
    if not _nested_join._native_library_available():
        raise RuntimeError("build the native extension before benchmarking")
    case = make_case(rows, scenario)
    expected = run_strategy(case, "fallback")
    for strategy in ["native", "explode"]:
        assert_frame_equal(run_strategy(case, strategy), expected)
    results = []
    for strategy in ["native", "fallback", "explode"]:
        command = [
            sys.executable,
            "-m",
            "performance.native_plugins.nested_join.benchmark",
            "--worker",
            strategy,
            "--rows",
            str(rows),
            "--scenario",
            scenario,
            "--repeats",
            str(repeats),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603
        results.append(json.loads(completed.stdout))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, default=int(os.environ.get("POLARS_LIST_MATH_PERF_ROWS", "2000"))
    )
    parser.add_argument(
        "--repeats", type=int, default=int(os.environ.get("POLARS_LIST_MATH_PERF_REPEATS", "3"))
    )
    parser.add_argument("--scenario", choices=["serp", "dialog", "skew"], default="dialog")
    parser.add_argument("--worker", choices=["native", "fallback", "explode"])
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    if args.worker:
        print(json.dumps(worker(args.rows, args.scenario, args.worker, args.repeats)))
    else:
        print(json.dumps(benchmark(args.rows, args.scenario, args.repeats), indent=2))

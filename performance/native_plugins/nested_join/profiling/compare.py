"""Compare independently compiled lookup libraries on the same resident inputs."""

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import polars as pl
from performance.native_plugins.nested_join.benchmark import (
    explode_join_implode,
    make_case,
    run_strategy,
)
from polars.testing import assert_frame_equal
from polars_list_math import _nested_join


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variants", nargs="+", help="Library subdirectory names, or 'explode'")
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--scenario", choices=("serp", "dialog", "skew"), default="skew")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.rows < 1 or args.repeats < 1:
        parser.error("rows and repeats must be positive")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variant names must be unique")
    libraries = {
        name: (args.library_root / name).resolve() for name in args.variants if name != "explode"
    }
    for name, library in libraries.items():
        if not library.is_dir() or not any(
            path.suffix in (".so", ".pyd", ".dll", ".dylib") for path in library.iterdir()
        ):
            parser.error(f"missing compiled plugin for {name}: {library}")

    case = make_case(args.rows, args.scenario)
    durations: dict[str, list[float]] = {name: [] for name in args.variants}
    expected = None
    for iteration in range(args.repeats + 1):
        order = args.variants if iteration % 2 else list(reversed(args.variants))
        for name in order:
            gc.collect()
            print(
                "PLM_RUN " + json.dumps({"variant": name, "iteration": iteration}),
                file=sys.stderr,
                flush=True,
            )
            # Build a fresh expression/index for each call. Each directory must
            # contain its own immutable binary: the plugin loader caches paths.
            with patch.object(_nested_join, "_LIB", libraries.get(name)):
                start = perf_counter()
                result = run_strategy(case, "explode" if name == "explode" else "native")
                elapsed = (perf_counter() - start) * 1000
            if iteration:
                durations[name].append(elapsed)
            else:
                if expected is None:
                    expected = explode_join_implode(case)
                assert_frame_equal(result, expected)
            del result

    print(
        json.dumps(
            {
                "rows": args.rows,
                "scenario": args.scenario,
                "leaves": case.leaves,
                "right_rows": case.right.height,
                "threads": pl.thread_pool_size(),
                "verified": True,
                "results": {
                    name: {
                        "wall_ms": values,
                        "min_ms": min(values),
                        "median_ms": statistics.median(values),
                    }
                    for name, values in durations.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

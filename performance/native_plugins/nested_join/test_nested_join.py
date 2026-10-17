import os

import pytest
from performance.native_plugins.nested_join.benchmark import benchmark
from polars_list_math import _nested_join


@pytest.mark.parametrize("scenario", ["serp", "dialog", "skew"])
def test_nested_join_performance(scenario):
    if not _nested_join._native_library_available():
        pytest.skip("build the native extension first")
    results = benchmark(
        int(os.environ.get("POLARS_LIST_MATH_PERF_ROWS", "2000")),
        scenario,
        int(os.environ.get("POLARS_LIST_MATH_PERF_REPEATS", "3")),
    )
    for result in results:
        print(result)

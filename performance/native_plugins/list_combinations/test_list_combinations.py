import polars as pl
import polars_list_math as plm
import pytest
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)

FRAME = pl.DataFrame(
    {
        "left": [
            [index % 17, (index + 1) % 17, None, (index + 3) % 17] for index in range(ROW_COUNT)
        ],
        "right": [[index % 17, (index + 2) % 17, (index + 4) % 17] for index in range(ROW_COUNT)],
    }
)
MODULE = plugin_module("_list_combinations")
CASES = (
    PerformanceCase(
        "list_combinations",
        MODULE,
        FRAME,
        lambda: plm.list_combinations("left", with_index=True, skip_null=True),
    ),
    PerformanceCase(
        "list_combinations_to",
        MODULE,
        FRAME,
        lambda: plm.list_combinations_to("left", "right", skip_null=True),
    ),
)

COMPLEX_FRAME = pl.DataFrame(
    {
        "left": [
            [None if value % 7 == 0 else (index + value) % 23 for value in range(16)]
            for index in range(ROW_COUNT)
        ],
        "right": [
            [None if value % 5 == 0 else (index * 2 + value) % 29 for value in range(12)]
            for index in range(ROW_COUNT)
        ],
    }
)
COMPLEX_CASES = (
    PerformanceCase(
        "list_combinations_complex",
        MODULE,
        COMPLEX_FRAME,
        lambda: plm.list_combinations("left", with_index=True, skip_null=True),
    ),
    PerformanceCase(
        "list_combinations_to_complex",
        MODULE,
        COMPLEX_FRAME,
        lambda: plm.list_combinations_to("left", "right", with_index=True, skip_null=True),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_list_combinations_performance(case: PerformanceCase) -> None:
    run_case(case)


@pytest.mark.parametrize("case", COMPLEX_CASES, ids=lambda case: case.name)
def test_list_combinations_complex_performance(case: PerformanceCase) -> None:
    run_case(case)

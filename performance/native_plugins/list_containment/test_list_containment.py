import polars as pl
import polars_list_math as plm
import pytest
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)

FLAT_FRAME = pl.DataFrame(
    {
        "left": [
            [index % 17, (index + 1) % 17, None, (index + 3) % 17] for index in range(ROW_COUNT)
        ],
        "right": [[index % 17, (index + 2) % 17, (index + 4) % 17] for index in range(ROW_COUNT)],
    }
)
NESTED_FRAME = pl.DataFrame(
    {
        "lists": [
            [[index % 13, 1, 2], [index % 13, 2, 3], [4, 5], [index % 13, 1]]
            for index in range(ROW_COUNT)
        ],
        "references": [[[index % 13, 1], [2, 3, 4], [index % 7, 5]] for index in range(ROW_COUNT)],
    }
)
CASES = (
    PerformanceCase(
        "list_containment",
        plugin_module("_list_containment"),
        FLAT_FRAME,
        lambda: plm.list_containment("left", "right"),
    ),
    PerformanceCase(
        "list_mean_containment",
        plugin_module("_list_containment"),
        NESTED_FRAME,
        lambda: plm.list_mean_containment("lists"),
    ),
    PerformanceCase(
        "list_mean_containment_to",
        plugin_module("_list_containment"),
        NESTED_FRAME,
        lambda: plm.list_mean_containment_to("lists", "references"),
    ),
)

COMPLEX_FLAT_FRAME = pl.DataFrame(
    {
        "left": [
            [None if value % 11 == 0 else (index + value * 3) % 41 for value in range(32)]
            for index in range(ROW_COUNT)
        ],
        "right": [
            [None if value % 13 == 0 else (index + value * 5) % 43 for value in range(28)]
            for index in range(ROW_COUNT)
        ],
    }
)
COMPLEX_NESTED_FRAME = pl.DataFrame(
    {
        "lists": [
            [
                [(index + list_index * 3 + value) % 47 for value in range(12)]
                for list_index in range(8)
            ]
            for index in range(ROW_COUNT)
        ],
        "references": [
            [
                [(index + list_index * 5 + value * 2) % 53 for value in range(10)]
                for list_index in range(6)
            ]
            for index in range(ROW_COUNT)
        ],
    }
)
COMPLEX_CASES = (
    PerformanceCase(
        "list_containment_complex",
        plugin_module("_list_containment"),
        COMPLEX_FLAT_FRAME,
        lambda: plm.list_containment("left", "right"),
    ),
    PerformanceCase(
        "list_mean_containment_complex",
        plugin_module("_list_containment"),
        COMPLEX_NESTED_FRAME,
        lambda: plm.list_mean_containment("lists"),
    ),
    PerformanceCase(
        "list_mean_containment_to_complex",
        plugin_module("_list_containment"),
        COMPLEX_NESTED_FRAME,
        lambda: plm.list_mean_containment_to("lists", "references"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_list_containment_performance(case: PerformanceCase) -> None:
    run_case(case)


@pytest.mark.parametrize("case", COMPLEX_CASES, ids=lambda case: case.name)
def test_list_containment_complex_performance(case: PerformanceCase) -> None:
    run_case(case)

import polars as pl
import polars_list_math as plm
import pytest
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)

MODULE = plugin_module("_list_kendall_tau")


def _frames(size: int, groups: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    left = [list(range(index % 7, index % 7 + size)) for index in range(ROW_COUNT)]
    right = [values[::2] + values[1::2][::-1] + [None] for values in left]
    flat = pl.DataFrame({"left": left, "right": right})
    nested = pl.DataFrame(
        {
            "lists": [
                [values[offset:] + values[:offset] for offset in range(groups)] + [None, []]
                for values in left
            ],
            "references": [[values, values[::-1], None] for values in right],
        }
    )
    return flat, nested


FLAT, NESTED = _frames(8, 3)
COMPLEX_FLAT, COMPLEX_NESTED = _frames(128, 8)
CASES = (
    PerformanceCase("kendall_tau", MODULE, FLAT, lambda: plm.list_kendall_tau("left", "right")),
    PerformanceCase("mean_kendall_tau", MODULE, NESTED, lambda: plm.list_mean_kendall_tau("lists")),
    PerformanceCase(
        "mean_kendall_tau_to",
        MODULE,
        NESTED,
        lambda: plm.list_mean_kendall_tau_to("lists", "references"),
    ),
    PerformanceCase(
        "kendall_tau_complex",
        MODULE,
        COMPLEX_FLAT,
        lambda: plm.list_kendall_tau("left", "right", normalize=True),
    ),
    PerformanceCase(
        "mean_kendall_tau_complex",
        MODULE,
        COMPLEX_NESTED,
        lambda: plm.list_mean_kendall_tau("lists", normalize=True),
    ),
    PerformanceCase(
        "mean_kendall_tau_to_complex",
        MODULE,
        COMPLEX_NESTED,
        lambda: plm.list_mean_kendall_tau_to("lists", "references", normalize=True),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_kendall_tau_performance(case: PerformanceCase) -> None:
    run_case(case)

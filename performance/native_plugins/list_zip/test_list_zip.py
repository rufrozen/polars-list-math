import polars as pl
import polars_list_math as plm
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)


def test_list_zip_performance() -> None:
    frame = pl.DataFrame(
        {
            "left": [[index, index + 1, None, index + 3] for index in range(ROW_COUNT)],
            "right": [[index, index + 2, index + 4] for index in range(ROW_COUNT)],
        }
    )
    run_case(
        PerformanceCase(
            "list_zip",
            plugin_module("_list_zip"),
            frame,
            lambda: plm.list_zip("left", "right", pad=True),
        )
    )


def test_list_zip_complex_performance() -> None:
    frame = pl.DataFrame(
        {
            "integers": [list(range(index, index + 24)) for index in range(ROW_COUNT)],
            "floats": [
                [value / 3 for value in range(index, index + 20)] for index in range(ROW_COUNT)
            ],
            "strings": [
                [f"row-{index}-value-{value}" for value in range(16)] for index in range(ROW_COUNT)
            ],
            "booleans": [[value % 3 == 0 for value in range(12)] for _ in range(ROW_COUNT)],
        }
    )
    run_case(
        PerformanceCase(
            "list_zip_complex",
            plugin_module("_list_zip"),
            frame,
            lambda: plm.list_zip("integers", "floats", "strings", "booleans", pad=True),
        )
    )

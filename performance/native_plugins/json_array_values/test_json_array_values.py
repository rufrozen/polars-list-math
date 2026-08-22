import json

import polars as pl
import polars_list_math as plm
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)


def test_json_array_values_performance() -> None:
    frame = pl.DataFrame(
        {
            "json": [
                f'["item {index}",{index},true,null,{{"nested":1}}]' for index in range(ROW_COUNT)
            ]
        }
    )
    run_case(
        PerformanceCase(
            "json_array_values",
            plugin_module("_json_array_values"),
            frame,
            lambda: plm.json_array_values("json"),
        )
    )


def test_json_array_values_complex_performance() -> None:
    values = [
        json.dumps(
            [
                {
                    "id": value,
                    "name": f"элемент {index}-{value}",
                    "values": list(range(value, value + 12)),
                    "metadata": {"active": value % 2 == 0, "missing": None},
                }
                for value in range(24)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for index in range(ROW_COUNT)
    ]
    frame = pl.DataFrame({"json": values})
    run_case(
        PerformanceCase(
            "json_array_values_complex",
            plugin_module("_json_array_values"),
            frame,
            lambda: plm.json_array_values("json"),
        )
    )

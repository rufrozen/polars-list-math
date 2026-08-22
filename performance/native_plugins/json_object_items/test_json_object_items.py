import json

import polars as pl
import polars_list_math as plm
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)


def test_json_object_items_performance() -> None:
    frame = pl.DataFrame(
        {
            "json": [
                f'{{"id":{index},"name":"item {index}","active":true,"tags":[1,2,3]}}'
                for index in range(ROW_COUNT)
            ]
        }
    )
    run_case(
        PerformanceCase(
            "json_object_items",
            plugin_module("_json_object_items"),
            frame,
            lambda: plm.json_object_items("json"),
        )
    )


def test_json_object_items_complex_performance() -> None:
    values = [
        json.dumps(
            {
                "id": index,
                "title": f"товар {index} & special / symbols",
                "active": index % 2 == 0,
                "metadata": {
                    "dimensions": {"width": index * 3, "height": index * 7},
                    "labels": [f"label-{value}" for value in range(12)],
                    "nullable": None,
                },
                "items": [
                    {"position": value, "score": value * 11, "enabled": value % 2 == 0}
                    for value in range(20)
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for index in range(ROW_COUNT)
    ]
    frame = pl.DataFrame({"json": values})
    run_case(
        PerformanceCase(
            "json_object_items_complex",
            plugin_module("_json_object_items"),
            frame,
            lambda: plm.json_object_items("json"),
        )
    )

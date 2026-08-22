import polars as pl
import polars_list_math as plm
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)


def test_url_query_encode_performance() -> None:
    frame = pl.DataFrame(
        {
            "query": [
                {"search": f"item {index}", "page": str(index % 10), "lang": "ru"}
                for index in range(ROW_COUNT)
            ]
        }
    )
    run_case(
        PerformanceCase(
            "url_query_encode",
            plugin_module("_url_query_encode"),
            frame,
            lambda: plm.url_query_encode("query"),
        )
    )


def test_url_query_encode_complex_performance() -> None:
    frame = pl.DataFrame(
        {
            "search": [f"товар {index} & category/special?" for index in range(ROW_COUNT)],
            "tags": [[f"tag {value}" for value in range(12)] for _ in range(ROW_COUNT)],
            "page": [index % 100 for index in range(ROW_COUNT)],
            "active": [index % 2 == 0 for index in range(ROW_COUNT)],
            "redirect": [
                f"https://example.com/items/{index}?source=тест#result"
                for index in range(ROW_COUNT)
            ],
        }
    )
    run_case(
        PerformanceCase(
            "url_query_encode_complex",
            plugin_module("_url_query_encode"),
            frame,
            lambda: plm.url_query_encode(
                pl.struct("search", "tags", "page", "active", "redirect"),
                doseq=True,
            ),
        )
    )

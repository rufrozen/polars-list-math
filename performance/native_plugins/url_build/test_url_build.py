import polars as pl
import polars_list_math as plm
from performance.native_plugins._benchmark import (
    ROW_COUNT,
    PerformanceCase,
    plugin_module,
    run_case,
)


def test_url_build_performance() -> None:
    frame = pl.DataFrame(
        {
            "scheme": ["https"] * ROW_COUNT,
            "netloc": ["example.com"] * ROW_COUNT,
            "path": [f"/items/{index}" for index in range(ROW_COUNT)],
            "params": [""] * ROW_COUNT,
            "query": [f"page={index % 10}&lang=ru" for index in range(ROW_COUNT)],
            "fragment": ["results"] * ROW_COUNT,
        }
    )
    run_case(
        PerformanceCase(
            "url_build",
            plugin_module("_url_build"),
            frame,
            lambda: plm.url_build(
                scheme="scheme",
                netloc="netloc",
                path="path",
                params="params",
                query="query",
                fragment="fragment",
            ),
        )
    )


def test_url_build_complex_performance() -> None:
    frame = pl.DataFrame(
        {
            "scheme": ["https"] * ROW_COUNT,
            "netloc": [
                f"user:password@subdomain-{index % 20}.example.com:8443"
                for index in range(ROW_COUNT)
            ],
            "path": [
                f"/каталог/category-{index % 30}/items/item-{index}/details"
                for index in range(ROW_COUNT)
            ],
            "params": [f"version={index % 5};mode=full" for index in range(ROW_COUNT)],
            "query": [
                f"search=item+{index}&tags=a&tags=b&redirect=%2Fitems%2F{index}"
                for index in range(ROW_COUNT)
            ],
            "fragment": [f"section-{index % 12}" for index in range(ROW_COUNT)],
        }
    )
    run_case(
        PerformanceCase(
            "url_build_complex",
            plugin_module("_url_build"),
            frame,
            lambda: plm.url_build(
                scheme="scheme",
                netloc="netloc",
                path="path",
                params="params",
                query="query",
                fragment="fragment",
            ),
        )
    )

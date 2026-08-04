"""Build URLs from optional component expressions."""

import polars as pl
from polars_list_math import url_build, url_query_encode


def main() -> None:
    frame = pl.DataFrame(
        {
            "host": ["example.com", "api.example.com"],
            "path": ["/search", "/items"],
            "term": ["rust polars", "book & pen"],
            "page": [1, 2],
            "fragment": ["results", None],
        }
    )

    query = url_query_encode(pl.struct(q=pl.col("term"), page=pl.col("page")))
    result = frame.select(
        url_build(
            scheme=pl.lit("https"),
            netloc="host",
            path="path",
            query=query,
            fragment="fragment",
        ).alias("url")
    )
    print(result)


if __name__ == "__main__":
    main()

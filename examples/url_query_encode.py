"""Encode Struct and key/value-list expressions as URL query strings."""

import polars as pl
import polars_list_math  # noqa: F401
from polars_list_math import url_query_encode


def main() -> None:
    frame = pl.DataFrame(
        {
            "name": ["Ada Lovelace"],
            "page": [2],
            "tags": [["python", "polars"]],
            "json_params": ['{"sort":"name asc","active":true}'],
        }
    )

    result = frame.select(
        url_query_encode(pl.struct("name", "page")).alias("query"),
        url_query_encode(pl.struct("name", "tags"), doseq=True).alias("query_with_tags"),
        url_query_encode(pl.col("json_params").str.json_object_items()).alias("query_from_json"),
    )
    print(result)


if __name__ == "__main__":
    main()

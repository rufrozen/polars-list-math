# pyright: strict

import polars as pl
from polars_list_math import (
    json_array_values,
    json_object_items,
    list_combinations,
    list_combinations_to,
    list_mean_similarity,
    list_mean_similarity_to,
    list_similarity,
    list_zip,
    url_build,
    url_query_encode,
)

items = pl.col("items")
other = pl.col("other")
nested = pl.col("nested")
json = pl.col("json")

expressions: list[pl.Expr] = [
    json_object_items(json, strict=True),
    json_array_values(json),
    list_zip(items, other, fields=["item", "other"]),
    list_combinations(items, skip_self=True),
    list_combinations_to(items, other),
    list_similarity(items, other, p=0.9),
    list_mean_similarity(nested),
    list_mean_similarity_to(nested, pl.col("reference")),
    url_query_encode(
        pl.struct("name", "page"),  # pyright: ignore[reportUnknownMemberType]
        doseq=True,
    ),
    url_build(
        scheme=pl.lit("https"),
        netloc=pl.col("host"),
        path=pl.col("path"),
    ),
]

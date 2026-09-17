# pyright: strict

import polars as pl
from polars_list_math import (
    ZipBuilder,
    expected_value_of_game,
    json_array_values,
    json_object_items,
    list_combinations,
    list_combinations_to,
    list_containment,
    list_mean_containment,
    list_mean_containment_to,
    list_mean_similarity,
    list_mean_similarity_to,
    list_similarity,
    list_zip,
    py_list_containment,
    url_build,
    url_query_encode,
)

items = pl.col("items")
other = pl.col("other")
nested = pl.col("nested")
json = pl.col("json")

expressions: list[pl.Expr] = [
    expected_value_of_game(nested),
    json_object_items(json, strict=True),
    json_array_values(json),
    list_zip(items, other, fields=["item", "other"]),
    list_zip(items, pl.lit("RUB"), broadcast=True, fields=["item", "currency"]),
    list_combinations(items, skip_self=True),
    list_combinations_to(items, other),
    list_containment(items, other),
    list_mean_containment(nested),
    list_mean_containment_to(nested, "reference"),
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

containment: float = py_list_containment([1, 2], [1, 2, 3])

zip_expression: pl.Expr = (
    ZipBuilder("a", "b", pad=True)
    .level(fields=["left", "right"])
    .filter(pl.element().struct.field("left").is_not_null())
    .eval(pl.element().struct.field("left") + pl.element().struct.field("right"))
    .build("nested_zip")
)

# pyright: strict

import polars as pl
from polars_list_math import (
    ZipBuilder,
    apply_schema,
    clean_schema,
    expected_value_of_game,
    find_null_schema,
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
    nested_join,
    nested_lookup,
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

lazy_frame = pl.LazyFrame({"value": [1, None], "missing": [None, None]})
null_schema: pl.Schema = find_null_schema(lazy_frame)
cleaned_schema: pl.Schema = clean_schema(lazy_frame.collect_schema(), null_schema)
cleaned_frame: pl.LazyFrame = apply_schema(lazy_frame, cleaned_schema)
projected_schema: pl.Schema = clean_schema(
    lazy_frame.collect_schema(), subschema=pl.Schema({"value": pl.Int64})
)

zip_expression: pl.Expr = (
    ZipBuilder("a", "b", pad=True)
    .level(fields=["left", "right"])
    .filter(pl.element().struct.field("left").is_not_null())
    .eval(pl.element().struct.field("left") + pl.element().struct.field("right"))
    .build("nested_zip")
)

join_left = pl.DataFrame({"sku": [[1, 2]]})
join_right = pl.DataFrame({"sku": [1], "label": [3]})
joined_eager: pl.DataFrame = nested_join(
    join_left, join_right, on={"sku": "sku"}, shape="sku", values={"label": "label"}
)
joined_lazy: pl.LazyFrame = nested_join(
    join_left, join_right.lazy(), on={"sku": "sku"}, shape="sku", values={"label": "label"}
)
joined_lazy_left: pl.LazyFrame = nested_join(
    join_left.lazy(), join_right, on={"sku": "sku"}, shape="sku", values={"label": "label"}
)
lookup_expression: pl.Expr = nested_lookup(
    "dictionary", on={"sku": "sku"}, shape="sku", values={"label": "label"}
)

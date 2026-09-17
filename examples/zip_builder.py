"""Build a nested zip with transformations at each level."""

import polars as pl
from polars_list_math import ZipBuilder

frame = pl.DataFrame(
    {
        "products": [[["apple", "pear"], ["tea"]]],
        "prices": [[[120, 90], [250]]],
        "currency": ["RUB"],
    }
)
element = pl.element()
expr = (
    ZipBuilder("products", "prices", "currency", broadcast=True)
    .filter(element.struct.field("products").list.len() > 0)
    .level(broadcast=True, fields=["product", "price", "currency"])
    .filter(element.struct.field("price") >= 100)
    .eval(
        pl.struct(
            element.struct.field("product"),
            (element.struct.field("price") * 0.9).alias("discounted_price"),
            element.struct.field("currency"),
        )
    )
    .build("offers")
)
print(frame.lazy().select(expr).collect())

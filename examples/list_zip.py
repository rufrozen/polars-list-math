"""Zip list columns into lists of structs."""

import polars as pl
import polars_list_math  # noqa: F401


def main() -> None:
    frame = pl.DataFrame(
        {
            "product": [["apple", "pear"], ["tea"]],
            "price": [[120, 90], [250, 300]],
            "currency": [["RUB", "RUB"], ["RUB", "RUB"]],
        }
    )

    result = frame.select(
        pl.col("product")
        .list.zip("price", "currency", fields=["product", "price", "currency"])
        .alias("offers"),
        pl.col("product").list.zip("price", pad=True).alias("padded_offers"),
    )
    print(result)


if __name__ == "__main__":
    main()

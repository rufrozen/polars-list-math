"""Zip list columns into lists of structs."""

import polars as pl
from polars_list_math import list_zip


def main() -> None:
    frame = pl.DataFrame(
        {
            "product": [["apple", "pear"], ["tea"]],
            "price": [[120, 90], [250, 300]],
            "currency": [["RUB", "RUB"], ["RUB", "RUB"]],
        }
    )

    result = frame.select(
        list_zip("product", "price", "currency", fields=["product", "price", "currency"]).alias(
            "offers"
        ),
        list_zip("product", "price", pad=True).alias("padded_offers"),
    )
    print(result)


if __name__ == "__main__":
    main()

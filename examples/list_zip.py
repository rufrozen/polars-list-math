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

    # A scalar column is repeated within each row's list.
    with_currency = frame.with_columns(pl.lit("RUB").alias("currency"))
    print(
        with_currency.select(
            list_zip("product", "price", "currency", broadcast=True).alias("offers"),
            # Only list lengths control padding; the scalar fills every element.
            list_zip("product", "price", "currency", broadcast=True, pad=True).alias("padded"),
        )
    )

    # Constants also work. Strings passed directly are column names, so use lit.
    print(
        frame.select(
            list_zip(
                "product",
                pl.lit("RUB"),
                pl.lit(None),
                broadcast=True,
                fields=["product", "currency", "discount"],
            ).alias("offers"),
        )
    )


if __name__ == "__main__":
    main()

"""Build pairs within a list and between two list columns."""

import polars as pl
import polars_list_math  # noqa: F401


def main() -> None:
    frame = pl.DataFrame(
        {
            "items": [["apple", "pear", "plum"]],
            "users": [[1, 2]],
            "products": [["book", "pen"]],
        }
    )

    result = frame.select(
        pl.col("items")
        .list.combinations(
            skip_self=True,
            left_value="first",
            right_value="second",
            left_index="first_index",
            right_index="second_index",
        )
        .alias("item_pairs"),
        pl.col("users")
        .list.combinations_to(
            "products",
            left_value="user_id",
            right_value="product",
        )
        .alias("user_product_pairs"),
    )
    print(result)


if __name__ == "__main__":
    main()

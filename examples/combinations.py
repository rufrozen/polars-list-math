"""Build pairs within a list and between two list columns."""

import polars as pl
from polars_list_math import list_combinations, list_combinations_to


def main() -> None:
    frame = pl.DataFrame(
        {
            "items": [["apple", "pear", "plum"]],
            "users": [[1, 2]],
            "products": [["book", "pen"]],
        }
    )

    result = frame.select(
        list_combinations(
            "items",
            skip_self=True,
            left_value="first",
            right_value="second",
            left_index="first_index",
            right_index="second_index",
        ).alias("item_pairs"),
        list_combinations_to(
            "users",
            "products",
            left_value="user_id",
            right_value="product",
        ).alias("user_product_pairs"),
    )
    print(result)


if __name__ == "__main__":
    main()

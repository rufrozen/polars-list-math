"""Convert JSON arrays into lists of string values."""

import polars as pl
import polars_list_math  # noqa: F401


def main() -> None:
    frame = pl.DataFrame(
        {
            "payload": [
                '["python",42,true,null,{"language":"rust"}]',
                '{"not":"an array"}',
                "invalid JSON",
            ]
        }
    )

    result = frame.with_columns(
        pl.col("payload").str.json_array_values().alias("values"),
    )
    print(result)

    # Nested arrays and objects become compact JSON strings.
    nested = pl.select(
        pl.lit('[[1,2],{"enabled":false}]').str.json_array_values().alias("nested_values"),
    )
    print(nested)


if __name__ == "__main__":
    main()

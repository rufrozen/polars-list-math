"""Convert JSON arrays into lists of string values."""

import polars as pl
from polars_list_math import json_array_values


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
        json_array_values("payload").alias("values"),
    )
    print(result)

    # Nested arrays and objects become compact JSON strings.
    nested = pl.select(
        json_array_values(pl.lit('[[1,2],{"enabled":false}]')).alias("nested_values"),
    )
    print(nested)


if __name__ == "__main__":
    main()

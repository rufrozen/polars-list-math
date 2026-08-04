"""Convert JSON objects into lists of key/value structs."""

import polars as pl
import polars_list_math  # noqa: F401


def main() -> None:
    frame = pl.DataFrame(
        {
            "payload": [
                '{"name":"Ada","active":true,"score":10,"note":null}',
                '["not", "an", "object"]',
                "invalid JSON",
            ]
        }
    )

    result = frame.with_columns(
        pl.col("payload").str.json_object_items().alias("items"),
    )
    print(result)

    # Invalid JSON raises instead of producing null in strict mode.
    valid = frame.head(1).select(
        pl.col("payload").str.json_object_items(strict=True).alias("strict_items"),
    )
    print(valid)


if __name__ == "__main__":
    main()

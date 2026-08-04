"""Convert JSON objects into lists of key/value structs."""

import polars as pl
from polars_list_math import json_object_items


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
        json_object_items("payload").alias("items"),
    )
    print(result)

    # Invalid JSON raises instead of producing null in strict mode.
    valid = frame.head(1).select(
        json_object_items("payload", strict=True).alias("strict_items"),
    )
    print(valid)


if __name__ == "__main__":
    main()

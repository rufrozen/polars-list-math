"""Nested lookup with row-level and ancestor keys."""

import polars as pl
from polars_list_math import nested_join, nested_lookup

left = pl.DataFrame({"query": ["phone"], "sku": [[[30, 10], [], [20, 30]]]})
right = pl.DataFrame(
    {"query": ["phone", "phone"], "sku": [10, 30], "grade": [1, 3]},
    schema_overrides={"grade": pl.Int8},
)

print(
    nested_join(
        left, right, on={"query": "query", "sku": "sku"}, shape="sku", values={"grade": "label"}
    )
)

print(
    left.lazy()
    .select(
        nested_lookup(
            pl.lit(right.to_struct().implode()),
            on={"query": "query", "sku": "sku"},
            shape="sku",
            values={"grade": "label"},
        )
    )
    .collect()
)

dialogs = pl.DataFrame(
    {
        "conversation": [["c1"]],
        "message": [[["m1"]]],
        "block": [[[[0, 1]]]],
        "sku": [[[[[7, 7], [7]]]]],
    }
)
labels = pl.DataFrame(
    {
        "conversation_id": ["c1", "c1"],
        "message_id": ["m1", "m1"],
        "block_id": [0, 1],
        "target_sku": [7, 7],
        "grade": [5, 8],
    }
)
print(
    nested_join(
        dialogs,
        labels,
        on={
            "conversation": "conversation_id",
            "message": "message_id",
            "block": "block_id",
            "sku": "target_sku",
        },
        shape="sku",
        values={"grade": "label"},
    )
)
# label: [[[[5, 5], [8]]]]

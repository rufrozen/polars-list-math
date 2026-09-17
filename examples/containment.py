"""Compare SERPs with directed, rank-independent containment."""

import polars as pl
from polars_list_math import (
    list_containment,
    list_mean_containment,
    list_mean_containment_to,
    py_list_containment,
)

print(py_list_containment(["a", "b"], ["a", "b", "c"]))

serps = pl.DataFrame({"a": [["a", "b"]], "b": [["a", "b", "c"]]})
print(
    serps.select(
        list_containment("a", "b").alias("a_in_b"),
        list_containment("b", "a").alias("b_in_a"),
    )
)

groups = pl.DataFrame(
    {
        "targets": [[["a", "b"], ["a", "b", "c"], ["d"]]],
        "references": [[["a", "b"], ["b"]]],
    }
)
print(
    groups.select(
        list_mean_containment("targets").alias("within_group"),
        list_mean_containment_to("targets", "references").alias("against_references"),
    )
)

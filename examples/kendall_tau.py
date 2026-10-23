"""Compare ranked ID lists with Kendall tau and optional normalization."""

import polars as pl
from polars_list_math import (
    list_kendall_tau,
    list_mean_kendall_tau,
    list_mean_kendall_tau_to,
    py_list_kendall_tau,
)


def main() -> None:
    frame = pl.DataFrame(
        {
            "expected": [["python", "rust", "sql"], ["python", "rust", "sql"]],
            "actual": [["rust", "python", "sql"], ["sql", "rust", "python"]],
        }
    )
    print(
        frame.with_columns(
            list_kendall_tau("expected", "actual").alias("tau"),
            list_kendall_tau("expected", "actual", normalize=True).alias("normalized"),
        )
    )

    nested = pl.DataFrame(
        {"lists": [[[1, 2, 3], [2, 1, 3], [3, 2, 1]]], "references": [[[1, 2, 3]]]}
    )
    print(
        nested.select(
            list_mean_kendall_tau("lists").alias("peer_mean"),
            list_mean_kendall_tau_to("lists", "references", normalize=True).alias("reference_mean"),
        )
    )
    print(py_list_kendall_tau([1, 2, 3], [2, 1, 3], normalize=True))


if __name__ == "__main__":
    main()

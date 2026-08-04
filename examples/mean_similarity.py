"""Calculate mean similarity for nested-list columns."""

import polars as pl
from polars_list_math import list_mean_similarity, list_mean_similarity_to


def main() -> None:
    frame = pl.DataFrame(
        {
            "recommendations": [[["python", "rust", "sql"], ["rust", "python", "sql"], ["java"]]],
            "references": [[["python", "rust", "sql"], ["python", "sql"]]],
        },
        schema={
            "recommendations": pl.List(pl.List(pl.String)),
            "references": pl.List(pl.List(pl.String)),
        },
    )

    result = frame.select(
        list_mean_similarity("recommendations").alias("similarity_within_recommendations"),
        list_mean_similarity_to("recommendations", "references").alias("similarity_to_references"),
    )
    print(result)


if __name__ == "__main__":
    main()

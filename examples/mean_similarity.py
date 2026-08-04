"""Calculate mean similarity for nested-list columns."""

import polars as pl
import polars_list_math  # noqa: F401


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
        pl.col("recommendations").list.mean_similarity().alias("similarity_within_recommendations"),
        pl.col("recommendations")
        .list.mean_similarity_to("references")
        .alias("similarity_to_references"),
    )
    print(result)


if __name__ == "__main__":
    main()

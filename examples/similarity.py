"""Compare ordered lists with weighted Jaccard similarity."""

import polars as pl
from polars_list_math import list_similarity, py_list_similarity


def main() -> None:
    frame = pl.DataFrame(
        {
            "expected": [["python", "rust", "sql"], ["python", "sql"]],
            "actual": [["rust", "python", "sql"], ["java", "sql"]],
        }
    )

    result = frame.with_columns(list_similarity("expected", "actual").alias("similarity"))
    print(result)

    score = py_list_similarity(
        ["python", "rust", "sql"],
        ["rust", "python", "sql"],
        p=0.9,
    )
    print(f"Python sequences: {score:.6f}")


if __name__ == "__main__":
    main()

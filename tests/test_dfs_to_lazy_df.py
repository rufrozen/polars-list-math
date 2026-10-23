import importlib
from collections.abc import Callable, Iterator
from typing import Literal

import polars as pl
import pytest
from polars_list_math import dfs_to_lazy_df

SCHEMA = pl.Schema({"id": pl.Int64, "name": pl.String, "score": pl.Float64})


def test_collects_multiple_batches_with_one_schema() -> None:
    batches = [
        pl.DataFrame({"id": [1, 2], "name": ["one", "two"], "score": [0.5, 0.7]}),
        pl.DataFrame({"id": [3], "name": ["three"], "score": [0.9]}),
    ]

    result = dfs_to_lazy_df(batches, SCHEMA).collect()

    assert result.schema == SCHEMA
    assert result.to_dict(as_series=False) == {
        "id": [1, 2, 3],
        "name": ["one", "two", "three"],
        "score": [0.5, 0.7, 0.9],
    }


def test_aligns_missing_extra_and_reordered_columns() -> None:
    batches = [
        pl.DataFrame({"score": [1], "id": [1], "extra": [True]}),
        pl.DataFrame({"name": ["two"], "id": [2], "score": [2]}),
        pl.DataFrame(
            {"id": ["3"], "name": ["three"], "score": ["3.0"]},
            schema={"id": pl.String, "name": pl.String, "score": pl.String},
        ),
    ]

    result = dfs_to_lazy_df(batches, SCHEMA).collect()

    assert result.schema == SCHEMA
    assert result.to_dict(as_series=False) == {
        "id": [1, 2, 3],
        "name": [None, "two", "three"],
        "score": [1.0, 2.0, 3.0],
    }


def test_applies_projection_and_predicate_pushdown() -> None:
    batches = [
        pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["one", "two", "three"],
                "score": [0.2, 0.8, 0.9],
            }
        )
    ]

    result = (
        dfs_to_lazy_df(batches, SCHEMA)
        .filter(pl.col("score") >= 0.8)
        .select("id", "name")
        .collect()
    )

    assert result.to_dict(as_series=False) == {
        "id": [2, 3],
        "name": ["two", "three"],
    }


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_explicitly_rejects_n_rows_pushdown(engine: Literal["streaming", "in-memory"]) -> None:
    lazy = dfs_to_lazy_df([pl.DataFrame({"id": [1], "name": ["one"], "score": [1.0]})], SCHEMA)

    with pytest.raises(pl.exceptions.ComputeError, match="n_rows pushdown is not supported"):
        lazy.head(1).collect(engine=engine)


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
@pytest.mark.parametrize("height", [0, 3])
@pytest.mark.parametrize("has_columns", [False, True])
def test_missing_columns_preserve_batch_height(
    engine: Literal["streaming", "in-memory"], height: int, has_columns: bool
) -> None:
    batch = pl.DataFrame({"extra": range(height)}) if has_columns else pl.DataFrame(height=height)
    result = dfs_to_lazy_df([batch], SCHEMA).collect(engine=engine)

    assert result.schema == SCHEMA
    assert result.height == height
    assert result.null_count().row(0) == (height, height, height)


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_zero_column_schema_preserves_batch_height(
    engine: Literal["streaming", "in-memory"],
) -> None:
    batches = [pl.DataFrame({"extra": [1, 2]}), pl.DataFrame(height=3)]

    result = dfs_to_lazy_df(batches, pl.Schema()).collect(engine=engine)

    assert result.shape == (5, 0)


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_row_count_with_empty_projection(engine: Literal["streaming", "in-memory"]) -> None:
    batches = [pl.DataFrame({"id": [1, 2]}), pl.DataFrame({"id": [3]})]
    lazy = dfs_to_lazy_df(batches, SCHEMA)

    assert lazy.select(pl.len()).collect(engine=engine).item() == 3
    assert lazy.filter(pl.col("id") > 1).select(pl.len()).collect(engine=engine).item() == 2


def test_honors_batch_size_with_non_copying_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("polars_list_math._dfs_to_lazy_df")
    sources: list[
        Callable[
            [list[str] | None, pl.Expr | None, int | None, int | None],
            Iterator[pl.DataFrame],
        ]
    ] = []

    def capture_source(
        io_source: Callable[
            [list[str] | None, pl.Expr | None, int | None, int | None],
            Iterator[pl.DataFrame],
        ],
        *,
        schema: pl.Schema,
    ) -> pl.LazyFrame:
        sources.append(io_source)
        return pl.LazyFrame(schema=schema)

    monkeypatch.setattr(module, "register_io_source", capture_source)
    module.dfs_to_lazy_df(
        [
            pl.DataFrame(
                {
                    "id": range(5),
                    "name": ["zero", "one", "two", "three", "four"],
                    "score": [0.0, 0.1, 0.2, 0.3, 0.4],
                }
            )
        ],
        SCHEMA,
    )

    batches = list(sources[0](["id"], pl.col("id") >= 0, None, 2))

    assert [batch.height for batch in batches] == [2, 2, 1]
    assert all(batch.columns == ["id"] for batch in batches)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(sources[0](None, None, None, 0))

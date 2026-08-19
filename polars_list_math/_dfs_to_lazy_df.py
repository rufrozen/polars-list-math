"""Create a lazy Polars IO source from an iterable of DataFrames."""

from collections.abc import Iterable, Iterator

import polars as pl
from polars.io.plugins import register_io_source


def dfs_to_lazy_df(dfs: Iterable[pl.DataFrame], schema: pl.Schema) -> pl.LazyFrame:
    """Expose an iterable of DataFrame batches as a lazy Polars IO source.

    Every batch is normalized to ``schema`` before it is yielded. Columns are
    reordered and cast to their declared dtypes, missing columns are filled
    with typed null values, and undeclared columns are discarded. Projection
    and predicate pushdown supplied by Polars are applied to each batch. If
    Polars supplies a ``batch_size`` hint, larger results are yielded as
    non-copying slices containing at most that many rows.

    Args:
        dfs: Re-iterable collection or one-shot iterator of DataFrame batches.
            A one-shot iterator, such as a generator, is consumed by the first
            execution. Collecting the returned LazyFrame again, or reusing it
            in multiple branches of one query, can therefore produce empty or
            incomplete results. Pass a re-iterable collection when the lazy
            plan may execute more than once.
        schema: Complete schema exposed by the lazy source. It also defines
            column order and the casts applied to mismatched batches.

    Returns:
        A LazyFrame backed by Polars' Python IO-plugin interface.

    Raises:
        NotImplementedError: Internally raised when Polars requests ``n_rows``
            pushdown. Polars wraps exceptions raised by Python IO sources in
            ``polars.exceptions.ComputeError`` during collection.

    Notes:
        ``head()``, ``limit()``, and equivalent operations that push a row
        limit into the source are not supported. Apply such operations only
        after materializing the LazyFrame.

        ``batch_size`` limits the size of each yielded DataFrame, not the total
        number of rows. Smaller input batches are not combined. The
        ``register_io_source`` function is an unstable Polars API, so
        compatibility should be checked when upgrading Polars.
    """

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        if n_rows is not None:
            raise NotImplementedError(
                "n_rows pushdown is not supported by dfs_to_lazy_df; "
                "head(), limit(), and equivalent lazy operations cannot be used"
            )
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive")

        for df in dfs:
            if df.schema != schema:
                df = _df_cast_schema(df, schema)

            # If the source supports predicate pushdown, the expression can be parsed
            # to skip rows/groups.
            if predicate is not None:
                df = df.filter(predicate)

            # If we would make a performant reader, we would not read these
            # columns at all.
            if with_columns is not None:
                df = df.select(with_columns)

            if batch_size is None or df.height <= batch_size:
                yield df
            else:
                yield from df.iter_slices(n_rows=batch_size)

    return register_io_source(io_source=source_generator, schema=schema)


def _df_cast_schema(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Select, add, order, and cast columns to the requested schema."""
    return df.select(
        pl.col(name).cast(dtype) if name in df.columns else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in schema.items()
    )

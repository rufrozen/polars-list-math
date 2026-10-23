"""Discover absent values throughout a lazy frame's nested schema."""

from collections.abc import Iterator

import polars as pl
import polars.selectors as cs
from polars.datatypes import DataTypeClass


def find_null_schema(lf: pl.LazyFrame) -> pl.Schema:
    """Return a nested schema containing only leaves with no non-null values.

    Inspect every row, including leaves inside arbitrarily nested List, Array,
    and Struct values. Scalar leaves without values are represented by Null,
    regardless of their declared dtype. Container types and array sizes are
    retained; struct fields and columns with values are omitted. A container
    with no populated leaves is therefore represented in full.

    Empty lists, empty structs, null parents, and an empty frame contribute no
    values. NaN, zero, False, and empty strings count as values.

    This operation executes one aggregate query for typed leaves and collects
    only one boolean per leaf, not the input frame. Dtype Null and empty
    structs need no data scan. The source must be replayable if the LazyFrame
    will subsequently be collected again (for example, after cleaning).
    """
    schema = lf.collect_schema()
    checks = [
        # Polars 2.0 rewrites is_not_null().any() to an aggregate that can
        # misread filtered empty chunks in the streaming engine. Keep the
        # row-level boolean evaluation before aggregation.
        check.fill_null(False).any().alias(str(index))
        for index, check in enumerate(
            check
            for name, dtype in schema.items()
            for check in _presence_checks(cs.by_name(name), dtype)
        )
    ]
    presence = iter(lf.select(checks).collect().row(0) if checks else ())
    result = pl.Schema()
    for name, dtype in schema.items():
        null_dtype = _null_dtype(dtype, presence)
        if null_dtype is not None:
            result[name] = null_dtype
    return result


def _presence_checks(expr: pl.Expr, dtype: pl.DataType | DataTypeClass) -> Iterator[pl.Expr]:
    """Build a row-level presence flag for each scalar leaf in schema order."""
    if dtype == pl.Null:
        return
    if isinstance(dtype, (pl.List, pl.Array)):
        values = expr.arr.to_list() if isinstance(dtype, pl.Array) else expr
        for check in _presence_checks(pl.element(), dtype.inner):
            yield values.list.eval(check).list.any().fill_null(False)
    elif isinstance(dtype, pl.Struct):
        # Struct lookup treats '*' as a selector even with positional access.
        # Temporary field names make all user-supplied names literal.
        children = expr.struct.rename_fields([str(index) for index in range(len(dtype.fields))])
        for index, field in enumerate(dtype.fields):
            for check in _presence_checks(children.struct.field(str(index)), field.dtype):
                # Parent validity also masks any physical child values.
                yield expr.is_not_null() & check
    else:
        yield expr.is_not_null()


def _null_dtype(dtype: pl.DataType | DataTypeClass, presence: Iterator[bool]) -> pl.DataType | None:
    """Rebuild only absent leaves from the ordered aggregate results."""
    if dtype == pl.Null:
        return pl.Null()
    if isinstance(dtype, (pl.List, pl.Array)):
        inner = _null_dtype(dtype.inner, presence)
        if inner is None:
            return None
        return pl.Array(inner, dtype.size) if isinstance(dtype, pl.Array) else pl.List(inner)
    if isinstance(dtype, pl.Struct):
        fields = []
        for field in dtype.fields:
            child = _null_dtype(field.dtype, presence)
            if child is not None:
                fields.append(pl.Field(field.name, child))
        return pl.Struct(fields) if fields or not dtype.fields else None
    return None if next(presence) else pl.Null()

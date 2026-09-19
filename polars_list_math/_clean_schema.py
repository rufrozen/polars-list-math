"""Subtract a subschema and apply the resulting schema lazily."""

import polars as pl
import polars.selectors as cs
from polars.datatypes import DataTypeClass


def clean_schema(schema: pl.Schema, subschema: pl.Schema) -> pl.Schema:
    """Remove columns and fields described by a subschema without reading data.

    Scalar leaves must match their source dtype; Null is also accepted as a
    marker removing any entire matching column or field. List, Array, and
    Struct descend into their matching containers. Structs may specify a subset
    of fields. Removing every field removes the struct and any List or Array
    wrappers around it, recursively up to the containing column.

    Preserve the order and dtypes of retained fields. Neither input is mutated.
    Unknown names, incompatible containers or array sizes, and mismatched
    scalar dtypes raise ValueError. The subschema can describe populated fields
    or come from find_null_schema; values do not affect removal.
    """
    schema, subschema = pl.Schema(schema), pl.Schema(subschema)
    _check_names(schema, subschema, "schema")
    result = pl.Schema()
    for name, dtype in schema.items():
        cleaned = _clean_dtype(dtype, subschema[name], repr(name)) if name in subschema else dtype
        if cleaned is not None:
            result[name] = cleaned
    return result


def apply_schema(lf: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
    """Select, order, and strictly cast columns to a schema, lazily.

    Polars casts nested containers recursively, dropping struct fields absent
    from the target dtype while preserving null parents, null list elements,
    empty lists, and array shapes. Typically ``schema`` comes from
    :func:`clean_schema`. No data is collected here. Missing top-level columns
    and invalid casts are reported by Polars when resolving/executing the plan;
    missing nested fields follow Polars cast semantics (filled with nulls).

    An empty target uses Polars' empty projection, yielding a 0-row, 0-column
    frame. Otherwise the row count is preserved.
    """
    return lf.select(cs.by_name(name).cast(dtype, strict=True) for name, dtype in schema.items())


def _check_names(schema: pl.Schema, subschema: pl.Schema, path: str) -> None:
    unknown = subschema.keys() - schema.keys()
    if unknown:
        raise ValueError(f"Unknown names at {path}: {sorted(unknown)!r}")


def _clean_dtype(
    dtype: pl.DataType | DataTypeClass, subschema: pl.DataType | DataTypeClass, path: str
) -> pl.DataType | None:
    if subschema == pl.Null:
        return None
    if isinstance(dtype, pl.Struct) and isinstance(subschema, pl.Struct):
        fields = pl.Schema({field.name: field.dtype for field in dtype.fields})
        subfields = pl.Schema({field.name: field.dtype for field in subschema.fields})
        _check_names(fields, subfields, path)
        kept = []
        for name, child in fields.items():
            cleaned = (
                _clean_dtype(child, subfields[name], f"{path}.{name!r}")
                if name in subfields
                else child
            )
            if cleaned is not None:
                kept.append(pl.Field(name, cleaned))
        return pl.Struct(kept) if kept else None
    if isinstance(dtype, pl.List) and isinstance(subschema, pl.List):
        inner = _clean_dtype(dtype.inner, subschema.inner, f"{path}[]")
        return pl.List(inner) if inner is not None else None
    if (
        isinstance(dtype, pl.Array)
        and isinstance(subschema, pl.Array)
        and dtype.size == subschema.size
    ):
        inner = _clean_dtype(dtype.inner, subschema.inner, f"{path}[]")
        return pl.Array(inner, dtype.size) if inner is not None else None
    if dtype == subschema:
        return None
    raise ValueError(f"Invalid subschema at {path}: {subschema!r} for {dtype!r}")

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, overload

import polars as pl
from polars._typing import PolarsDataType
from polars.plugins import register_plugin_function

_LIB = Path(__file__).parent
_EXTENSIONS = {".so", ".pyd", ".dll", ".dylib"}
_KEY_TYPES = {
    pl.Boolean,
    pl.String,
    pl.Binary,
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}


def _native_library_available() -> bool:
    return any(p.is_file() and p.suffix in _EXTENSIONS for p in _LIB.iterdir())


def _mapping(value: Mapping[str, str], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{name} must be a non-empty mapping of column names")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise TypeError(f"{name} must map strings to strings")
    return dict(value)


def _arguments(
    on: Mapping[str, str], shape: str, values: Mapping[str, str], skip_null_keys: bool
) -> tuple[dict[str, str], dict[str, str]]:
    on, values = _mapping(on, "on"), _mapping(values, "values")
    if not isinstance(shape, str):
        raise TypeError("shape must be a column name")
    if not isinstance(skip_null_keys, bool):
        raise TypeError("skip_null_keys must be a bool")
    if len(set(values.values())) != len(values):
        raise pl.exceptions.DuplicateError("output names must be unique")
    return on, values


def _depth(dtype: PolarsDataType) -> tuple[int, PolarsDataType]:
    depth = 0
    while isinstance(dtype, pl.List):
        depth += 1
        dtype = dtype.inner
    return depth, dtype


def _wrap(dtype: PolarsDataType, depth: int) -> PolarsDataType:
    for _ in range(depth):
        dtype = pl.List(dtype)
    return dtype


def _output_dtype(
    dtypes: Sequence[PolarsDataType], on: Mapping[str, str], values: Mapping[str, str]
) -> pl.Struct:
    right, shape, *keys = dtypes
    if not isinstance(right, pl.List) or not isinstance(right.inner, pl.Struct):
        raise pl.exceptions.SchemaError("right must have dtype List[Struct]")
    fields = {f.name: f.dtype for f in right.inner.fields}
    depth, _ = _depth(shape)
    if not depth:
        raise pl.exceptions.SchemaError("shape must have List dtype")
    for (left_name, right_name), dtype in zip(on.items(), keys, strict=True):
        if right_name not in fields:
            raise pl.exceptions.ColumnNotFoundError(right_name)
        key_depth, leaf = _depth(dtype)
        if key_depth > depth:
            raise pl.exceptions.SchemaError(f"key {left_name!r} is deeper than shape")
        if leaf not in _KEY_TYPES or fields[right_name] not in _KEY_TYPES:
            raise pl.exceptions.SchemaError(
                f"unsupported key dtype for {left_name!r}: {leaf} / {fields[right_name]}"
            )
        if leaf != fields[right_name]:
            raise pl.exceptions.SchemaError(
                f"key dtype mismatch for {left_name!r}: {leaf} != {fields[right_name]}"
            )
    for name in values:
        if name not in fields:
            raise pl.exceptions.ColumnNotFoundError(name)
    return pl.Struct({alias: _wrap(fields[name], depth) for name, alias in values.items()})


def nested_lookup(
    right: str | pl.Expr,
    *,
    on: Mapping[str, str],
    shape: str,
    values: Mapping[str, str],
    skip_null_keys: bool = False,
) -> pl.Expr:
    """Look up ancestor/leaf keys in a List[Struct] dictionary.

    ``on`` maps left column names to dictionary field names. ``values`` maps
    dictionary fields to output names. The result is a Struct of columns with
    the List structure of ``shape``. The dictionary can have one row (shared)
    or one row per left row. Null keys raise unless explicitly skipped; skipping
    keeps left positions and fills them with null values. Key dtypes must match.
    """
    on, values = _arguments(on, shape, values, skip_null_keys)
    if not isinstance(right, (str, pl.Expr)):
        raise TypeError("right must be a column name or expression")
    return _lookup_expression(
        right, on, shape, values, skip_null_keys, _native_library_available()
    ).alias("nested_lookup")


def _lookup_expression(
    right: str | pl.Expr,
    on: dict[str, str],
    shape: str,
    values: dict[str, str],
    skip_null_keys: bool,
    native: bool,
) -> pl.Expr:
    args = [right, pl.col(shape), *(pl.col(name) for name in on)]
    if native:
        return register_plugin_function(
            plugin_path=_LIB,
            function_name="nested_lookup",
            args=args,
            kwargs={
                "left_keys": list(on),
                "right_keys": list(on.values()),
                "value_fields": list(values),
                "output_names": list(values.values()),
                "skip_null_keys": skip_null_keys,
            },
            # A singleton right input must remain whole, including on empty left.
            is_elementwise=False,
            use_abs_path=True,
        )

    def dtype_probe(columns: Sequence[pl.Series]) -> pl.Series:
        # Only schema inference: never validate Polars' synthetic sample values.
        try:
            dtype = _output_dtype([s.dtype for s in columns], on, values)
        except pl.exceptions.PolarsError:
            # Polars replaces exceptions during UDF dtype inference with a
            # generic inference error. Defer invalid-schema diagnostics to the
            # real callback, which raises the original, actionable error.
            dtype = pl.Struct({name: pl.Null for name in values.values()})
        return pl.Series([], dtype=dtype)

    dtype = pl.dtype_of(pl.map_batches(args, dtype_probe))
    return pl.map_batches(
        args,
        partial(_lookup_batch, on=on, values=values, skip_null_keys=skip_null_keys),
        return_dtype=dtype,
    )


def _lookup_batch(
    columns: Sequence[pl.Series],
    *,
    on: dict[str, str],
    values: dict[str, str],
    skip_null_keys: bool,
) -> pl.Series:
    dtype = _output_dtype([s.dtype for s in columns], on, values)
    right, shape, *keys = columns
    size = len(shape)
    if len(right) not in (1, size) or any(len(key) not in (1, size) for key in keys):
        raise pl.exceptions.ShapeError("lookup inputs must match shape row count or have length 1")
    shape_depth = _depth(shape.dtype)[0]
    key_depths = [_depth(key.dtype)[0] for key in keys]
    key_data = [key.to_list() for key in keys]
    fields = list(values)

    def build_index(records: list[dict[str, Any]] | None) -> dict[tuple[Any, ...], Any]:
        if records is None:
            raise pl.exceptions.ComputeError("right dictionary must not be null; use an empty list")
        index: dict[tuple[Any, ...], Any] = {}
        for record in records:
            key = tuple(record[name] for name in on.values()) if record is not None else (None,)
            if any(part is None for part in key):
                if skip_null_keys:
                    continue
                raise pl.exceptions.ComputeError(f"null in right key: {key!r}")
            if key in index:
                detail = dict(zip(on.values(), key, strict=True))
                raise pl.exceptions.ComputeError(f"duplicate right key: {detail!r}")
            index[key] = tuple(record[name] for name in fields)
        return index

    # All supplied dictionaries are validated, even for an empty left input.
    indexes = [build_index(records) for records in right.to_list()]

    def walk(
        node: Any, depth: int, components: list[Any], index: dict[tuple[Any, ...], Any]
    ) -> list[Any]:
        if depth == shape_depth:
            if any(part is None for part in components):
                if not skip_null_keys:
                    raise pl.exceptions.ComputeError(f"null in left key: {components!r}")
                return [None] * len(fields)
            return list(index.get(tuple(components), (None,) * len(fields)))
        if node is None:
            return [None] * len(fields)
        for name, component, key_depth in zip(on, components, key_depths, strict=True):
            if key_depth > depth and (
                not isinstance(component, list) or len(component) != len(node)
            ):
                raise pl.exceptions.ShapeError(f"key {name!r} does not align with shape")
        output: list[list[Any]] = [[] for _ in fields]
        for i, child in enumerate(node):
            child_components = [
                component[i] if key_depth > depth else component
                for component, key_depth in zip(components, key_depths, strict=True)
            ]
            for destination, value in zip(
                output, walk(child, depth + 1, child_components, index), strict=True
            ):
                destination.append(value)
        return output

    rows = []
    for row, node in enumerate(shape.to_list()):
        components = [data[0 if len(data) == 1 else row] for data in key_data]
        result = walk(node, 0, components, indexes[0 if len(indexes) == 1 else row])
        rows.append(dict(zip(values.values(), result, strict=True)))
    return pl.Series("nested_lookup", rows, dtype=dtype)


@overload
def nested_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    on: Mapping[str, str],
    shape: str,
    values: Mapping[str, str],
    skip_null_keys: bool = False,
) -> pl.DataFrame: ...


@overload
def nested_join(
    left: pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    on: Mapping[str, str],
    shape: str,
    values: Mapping[str, str],
    skip_null_keys: bool = False,
) -> pl.LazyFrame: ...


@overload
def nested_join(
    left: pl.DataFrame,
    right: pl.LazyFrame,
    *,
    on: Mapping[str, str],
    shape: str,
    values: Mapping[str, str],
    skip_null_keys: bool = False,
) -> pl.LazyFrame: ...


def nested_join(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    on: Mapping[str, str],
    shape: str,
    values: Mapping[str, str],
    skip_null_keys: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Attach flat right values to nested left positions without exploding left.

    ``on`` maps left columns (at any ancestor depth) to right key columns;
    ``values`` maps right value columns to new left column names. ``shape``
    supplies List structure. Right keys must be unique and have matching dtypes.
    Float keys and output name collisions are errors. Null keys are errors by
    default; ``skip_null_keys=True`` ignores them while preserving left shape.

    Two DataFrames return a DataFrame. Any LazyFrame input returns a LazyFrame
    without collecting data. Execution materializes both inputs; it is not a
    streaming join. Full-right validation is retained even for empty left input.
    """
    on, values = _arguments(on, shape, values, skip_null_keys)
    if not isinstance(left, (pl.DataFrame, pl.LazyFrame)) or not isinstance(
        right, (pl.DataFrame, pl.LazyFrame)
    ):
        raise TypeError("left and right must be DataFrame or LazyFrame")
    left_schema, right_schema = left.collect_schema(), right.collect_schema()
    for name in [shape, *on]:
        if name not in left_schema:
            raise pl.exceptions.ColumnNotFoundError(name)
    for name in [*on.values(), *values]:
        if name not in right_schema:
            raise pl.exceptions.ColumnNotFoundError(name)
    collision = set(left_schema).intersection(values.values())
    if collision:
        raise pl.exceptions.DuplicateError(
            f"output names already exist in left: {sorted(collision)}"
        )
    needed = list(dict.fromkeys([*on.values(), *values]))
    packed_dtype = pl.List(pl.Struct({name: right_schema[name] for name in needed}))
    result_dtype = _output_dtype(
        [packed_dtype, left_schema[shape], *(left_schema[name] for name in on)], on, values
    )
    output_schema: dict[str, PolarsDataType] = dict(left_schema)
    output_schema.update({field.name: field.dtype for field in result_dtype.fields})
    # Capture the backend during construction, as with the other expression helpers.
    native = _native_library_available()

    def execute(frame: pl.DataFrame, dictionary: pl.Series) -> pl.DataFrame:
        expression = _lookup_expression(
            pl.lit(dictionary), on, shape, values, skip_null_keys, native
        )
        extra = frame.select(expression.alias("__lookup")).to_series().struct.unnest()
        return frame.hstack(extra)

    if isinstance(left, pl.DataFrame) and isinstance(right, pl.DataFrame):
        return execute(left, right.select(needed).to_struct().implode())

    # Pack each input into one row. The cross join pairs two singletons rather
    # than duplicating the right dictionary for every left row. Empty tables
    # still produce a singleton containing [], so right validation always runs.
    packed_left = left.lazy().select(pl.struct(pl.all()).implode().alias("__left"))
    packed_right = right.lazy().select(pl.struct(needed).implode().alias("__right"))

    def execute_packed(batch: pl.DataFrame) -> pl.DataFrame:
        frame = batch["__left"][0].struct.unnest()
        return execute(frame, batch["__right"])

    return packed_left.join(packed_right, how="cross").map_batches(
        execute_packed,
        schema=output_schema,
        predicate_pushdown=False,
        projection_pushdown=False,
        slice_pushdown=False,
        streamable=False,
    )

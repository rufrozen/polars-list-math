from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import NamedTuple

import polars as pl
import pytest

ROW_COUNT = 10_000
CHILD_DTYPE = pl.Struct({"code": pl.Int64, "name": pl.String})
SCHEMA = {
    "row_id": pl.Int64,
    "child": CHILD_DTYPE,
    "children": pl.List(CHILD_DTYPE),
}


class NamedTupleChild(NamedTuple):
    code: int
    name: str


@dataclass(slots=True)
class SlottedDataclassChild:
    code: int
    name: str


def expected_row(index: int) -> dict[str, object]:
    return {
        "row_id": index,
        "child": {"code": index, "name": f"child-{index}"},
        "children": [
            {"code": index * 2, "name": "first"},
            {"code": index * 2 + 1, "name": "second"},
        ],
    }


def dict_row(index: int) -> tuple[object, ...]:
    expected = expected_row(index)
    return expected["row_id"], expected["child"], expected["children"]


def namedtuple_row(index: int) -> tuple[object, ...]:
    return (
        index,
        NamedTupleChild(index, f"child-{index}"),
        [
            NamedTupleChild(index * 2, "first"),
            NamedTupleChild(index * 2 + 1, "second"),
        ],
    )


@pytest.mark.parametrize("row_factory", (dict_row, namedtuple_row))
@pytest.mark.parametrize("explicit_schema", (False, True))
def test_supported_nested_representations_preserve_data(
    row_factory: Callable[[int], tuple[object, ...]], explicit_schema: bool
) -> None:
    schema = SCHEMA if explicit_schema else list(SCHEMA)

    frame = pl.DataFrame([row_factory(3)], orient="row", schema=schema)

    assert frame.schema == pl.Schema(SCHEMA)
    assert frame.to_dicts() == [expected_row(3)]


def test_ordinary_tuples_cannot_represent_nested_structs() -> None:
    row = (3, (3, "child-3"), [(6, "first"), (7, "second")])

    with pytest.raises(TypeError, match="unexpected value while building Series"):
        pl.DataFrame([row], orient="row", schema=SCHEMA)


def test_slotted_dataclass_is_not_a_safe_nested_input() -> None:
    row = (
        3,
        SlottedDataclassChild(3, "child-3"),
        [SlottedDataclassChild(6, "first")],
    )

    frame = pl.DataFrame([row], orient="row", schema=SCHEMA)

    assert frame.to_dicts() != [expected_row(3)]
    assert frame.to_dicts() == [
        {"row_id": 3, "child": {"code": None, "name": None}, "children": None}
    ]


@pytest.mark.parametrize("row_factory", (dict_row, namedtuple_row))
def test_construction_timing(row_factory: Callable[[int], tuple[object, ...]]) -> None:
    rows = [row_factory(index) for index in range(ROW_COUNT)]

    started_at = perf_counter()
    frame = pl.DataFrame(rows, orient="row", schema=SCHEMA)
    elapsed = perf_counter() - started_at

    print(f"{row_factory.__name__:>18} rows={ROW_COUNT}: {elapsed:.4f}s")
    assert frame.shape == (ROW_COUNT, 3)
    assert frame.row(ROW_COUNT - 1, named=True) == expected_row(ROW_COUNT - 1)

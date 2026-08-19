from __future__ import annotations

import os
from dataclasses import dataclass, fields
from datetime import date, datetime, timezone
from time import perf_counter
from typing import NamedTuple

import polars as pl
import pytest

DEFAULT_ROW_COUNT = 10_000
ROW_COUNT = int(os.environ.get("POLARS_CONSTRUCTION_ROWS", DEFAULT_ROW_COUNT))


@dataclass(frozen=True)
class DataclassRow:
    row_id: int
    name: str
    active: bool
    score: float
    created_at: datetime
    birth_date: date
    count: int
    ratio: float
    category: str
    optional_note: str | None
    tags: list[str]
    values: list[int]
    weights: list[float]
    flags: list[bool]
    address: dict[str, object]
    events: list[dict[str, object]]
    metadata: dict[str, str | int]
    point: dict[str, float]
    aliases: list[str]
    measurements: list[float]
    status: str
    priority: int
    amount: float
    enabled: bool
    group_id: int
    description: str
    codes: list[int]
    properties: dict[str, str]
    history: list[dict[str, int | str]]
    version: int


@dataclass(frozen=True, slots=True)
class SlottedDataclassRow:
    row_id: int
    name: str
    active: bool
    score: float
    created_at: datetime
    birth_date: date
    count: int
    ratio: float
    category: str
    optional_note: str | None
    tags: list[str]
    values: list[int]
    weights: list[float]
    flags: list[bool]
    address: dict[str, object]
    events: list[dict[str, object]]
    metadata: dict[str, str | int]
    point: dict[str, float]
    aliases: list[str]
    measurements: list[float]
    status: str
    priority: int
    amount: float
    enabled: bool
    group_id: int
    description: str
    codes: list[int]
    properties: dict[str, str]
    history: list[dict[str, int | str]]
    version: int


class NamedTupleRow(NamedTuple):
    row_id: int
    name: str
    active: bool
    score: float
    created_at: datetime
    birth_date: date
    count: int
    ratio: float
    category: str
    optional_note: str | None
    tags: list[str]
    values: list[int]
    weights: list[float]
    flags: list[bool]
    address: dict[str, object]
    events: list[dict[str, object]]
    metadata: dict[str, str | int]
    point: dict[str, float]
    aliases: list[str]
    measurements: list[float]
    status: str
    priority: int
    amount: float
    enabled: bool
    group_id: int
    description: str
    codes: list[int]
    properties: dict[str, str]
    history: list[dict[str, int | str]]
    version: int


ROW_TYPES = (DataclassRow, SlottedDataclassRow, NamedTupleRow)
EXPECTED_COLUMNS = list(NamedTupleRow._fields)


def make_row_values(index: int) -> tuple[object, ...]:
    """Build deterministic values shared by all three row representations."""
    return (
        index,
        f"row-{index}",
        index % 2 == 0,
        index / 10,
        datetime(2025, 1, 1, 12, index % 60, tzinfo=timezone.utc),
        date(1990 + index % 30, 1 + index % 12, 1 + index % 28),
        index * 2,
        (index % 100) / 100,
        f"category-{index % 5}",
        None if index % 3 == 0 else f"note-{index}",
        [f"tag-{index % 3}", f"tag-{(index + 1) % 3}"],
        [index, index + 1, index + 2],
        [index / 10, (index + 1) / 10],
        [index % 2 == 0, index % 3 == 0],
        {
            "city": f"city-{index % 10}",
            "zip_code": 100_000 + index,
            "coordinates": {"lat": 55.75 + index / 100_000, "lon": 37.61},
        },
        [
            {
                "event_id": index * 2,
                "label": "created",
                "attributes": {"source": "generator", "rank": index},
            },
            {
                "event_id": index * 2 + 1,
                "label": "updated",
                "attributes": {"source": "generator", "rank": index + 1},
            },
        ],
        {"source": "synthetic", "partition": index % 4},
        {"x": float(index), "y": float(index + 1)},
        [f"alias-{index}", f"alias-{index + 1}"],
        [float(index), float(index) + 0.5],
        "ready" if index % 2 == 0 else "pending",
        index % 10,
        index * 1.25,
        index % 5 != 0,
        index % 100,
        f"generated row {index}",
        [index % 7, index % 11],
        {"owner": f"team-{index % 3}", "region": "eu"},
        [{"step": 1, "state": "new"}, {"step": 2, "state": "processed"}],
        1 + index % 3,
    )


def make_rows(row_type: type, count: int) -> list[object]:
    return [row_type(*make_row_values(index)) for index in range(count)]


def assert_frame(frame: pl.DataFrame, expected_height: int) -> None:
    assert frame.shape == (expected_height, 30)
    assert frame.columns == EXPECTED_COLUMNS
    assert isinstance(frame.schema["address"], pl.Struct)
    events_dtype = frame.schema["events"]
    assert isinstance(events_dtype, pl.List)
    assert isinstance(events_dtype.inner, pl.Struct)
    assert frame.schema["tags"] == pl.List(pl.String)
    assert frame["row_id"].to_list() == list(range(expected_height))


@pytest.mark.parametrize("row_type", ROW_TYPES, ids=lambda item: item.__name__)
@pytest.mark.parametrize("row_count", (1, ROW_COUNT), ids=("one_row", "n_rows"))
def test_dataframe_construction(row_type: type, row_count: int) -> None:
    rows = make_rows(row_type, row_count)

    started_at = perf_counter()
    frame = pl.DataFrame(rows, orient="row", schema=EXPECTED_COLUMNS)
    elapsed = perf_counter() - started_at

    print(f"{row_type.__name__:>20} rows={row_count:>7}: {elapsed:.4f}s")
    assert_frame(frame, row_count)


def test_all_representations_have_exactly_30_fields() -> None:
    assert len(fields(DataclassRow)) == 30
    assert len(fields(SlottedDataclassRow)) == 30
    assert len(NamedTupleRow._fields) == 30

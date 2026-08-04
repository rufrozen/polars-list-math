from __future__ import annotations

import polars as pl
import pytest
from polars_list_math import _json_object_items, json_object_items

EXPECTED_DTYPE = pl.List(
    pl.Struct(
        {
            "key": pl.String,
            "value": pl.String,
        }
    )
)


def test_json_object_items_converts_object_values() -> None:
    df = pl.DataFrame(
        {
            "json": [
                '{"text":"hello","number":12.5,"boolean":true,"null":null}',
                '{"array":[1,"x"],"object":{"nested":false}}',
                "{}",
            ]
        }
    )

    result = df.select(json_object_items("json").alias("items"))["items"]

    assert result.dtype == EXPECTED_DTYPE
    assert result.to_list() == [
        [
            {"key": "text", "value": "hello"},
            {"key": "number", "value": "12.5"},
            {"key": "boolean", "value": "true"},
            {"key": "null", "value": None},
        ],
        [
            {"key": "array", "value": '[1,"x"]'},
            {"key": "object", "value": '{"nested":false}'},
        ],
        [],
    ]


def test_json_object_items_string_namespace() -> None:
    df = pl.DataFrame({"json": ['{"name":"Ada","active":true}']})

    result = df.select(pl.col("json").str.json_object_items().alias("items"))["items"]

    assert result.dtype == EXPECTED_DTYPE
    assert result.to_list() == [
        [
            {"key": "name", "value": "Ada"},
            {"key": "active", "value": "true"},
        ]
    ]


def test_json_object_items_string_namespace_passes_strict() -> None:
    df = pl.DataFrame({"json": ["invalid"]})

    with pytest.raises(pl.exceptions.ComputeError, match="invalid JSON"):
        df.select(pl.col("json").str.json_object_items(strict=True))


def test_json_object_items_returns_null_for_null_non_object_and_invalid_json() -> None:
    df = pl.DataFrame(
        {
            "json": [
                None,
                "null",
                "[]",
                '"text"',
                "42",
                "true",
                "{invalid",
            ]
        },
        schema={"json": pl.String},
    )

    result = df.select(json_object_items(pl.col("json")))["json"]

    assert result.dtype == EXPECTED_DTYPE
    assert result.to_list() == [None] * 7


def test_json_object_items_preserves_order_and_unicode() -> None:
    df = pl.DataFrame({"json": ['{"z":"привет","a":"\\u263a"}']})

    result = df.select(json_object_items("json"))["json"]

    assert result.to_list() == [
        [
            {"key": "z", "value": "привет"},
            {"key": "a", "value": "☺"},
        ]
    ]


def test_json_object_items_strict_rejects_invalid_json() -> None:
    df = pl.DataFrame({"json": ["{}", "{invalid"]})

    with pytest.raises((pl.exceptions.ComputeError, ValueError), match="JSON|json|Expecting"):
        df.select(json_object_items("json", strict=True))


def test_json_object_items_validates_strict() -> None:
    with pytest.raises(TypeError, match="strict must be a bool"):
        json_object_items("json", strict="yes")  # type: ignore[arg-type]


def test_json_object_items_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_json_object_items, "_native_library_available", lambda: False)
    df = pl.DataFrame({"json": ['{"a":1,"b":null}', "invalid", "[]"]})

    result = df.select(json_object_items("json"))["json"]

    assert result.dtype == EXPECTED_DTYPE
    assert result.to_list() == [
        [{"key": "a", "value": "1"}, {"key": "b", "value": None}],
        None,
        None,
    ]

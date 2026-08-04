from __future__ import annotations

import polars as pl
import pytest
from polars_list_math import _json_array_values, json_array_values


def test_json_array_values_converts_values() -> None:
    df = pl.DataFrame(
        {
            "json": [
                '["hello",12.5,true,null]',
                '[[1,"x"],{"nested":false}]',
                "[]",
            ]
        }
    )

    result = df.select(pl.col("json").str.json_array_values().alias("values"))["values"]

    assert result.dtype == pl.List(pl.String)
    assert result.to_list() == [
        ["hello", "12.5", "true", None],
        ['[1,"x"]', '{"nested":false}'],
        [],
    ]


def test_json_array_values_top_level_function() -> None:
    df = pl.DataFrame({"json": ['["a",1]']})

    result = df.select(json_array_values("json"))["json"]

    assert result.to_list() == [["a", "1"]]


def test_json_array_values_returns_null_for_null_non_array_and_invalid_json() -> None:
    df = pl.DataFrame(
        {"json": [None, "null", "{}", '"text"', "42", "true", "[invalid"]},
        schema={"json": pl.String},
    )

    result = df.select(pl.col("json").str.json_array_values())["json"]

    assert result.dtype == pl.List(pl.String)
    assert result.to_list() == [None] * 7


def test_json_array_values_preserves_unicode_and_nested_object_order() -> None:
    df = pl.DataFrame({"json": ['["привет","\\u263a",{"z":1,"a":2}]']})

    result = df.select(pl.col("json").str.json_array_values())["json"]

    assert result.to_list() == [["привет", "☺", '{"z":1,"a":2}']]


def test_json_array_values_strict_rejects_invalid_json() -> None:
    df = pl.DataFrame({"json": ["[]", "[invalid"]})

    with pytest.raises(pl.exceptions.ComputeError, match="invalid JSON"):
        df.select(pl.col("json").str.json_array_values(strict=True))


def test_json_array_values_validates_strict() -> None:
    with pytest.raises(TypeError, match="strict must be a bool"):
        pl.col("json").str.json_array_values(strict="yes")  # type: ignore[arg-type]


def test_json_array_values_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_json_array_values, "_native_library_available", lambda: False)
    df = pl.DataFrame({"json": ['["a",1,null]', "invalid", "{}"]})

    result = df.select(pl.col("json").str.json_array_values())["json"]

    assert result.dtype == pl.List(pl.String)
    assert result.to_list() == [["a", "1", None], None, None]

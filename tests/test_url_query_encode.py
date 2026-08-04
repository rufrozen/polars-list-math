from __future__ import annotations

import polars as pl
import pytest
from polars_list_math import _url_query_encode, url_query_encode


def test_url_query_encode_struct() -> None:
    df = pl.DataFrame(
        {
            "name": ["Ada Lovelace", "Grace & Hopper"],
            "page": [2, 3],
            "active": [True, False],
            "missing": [None, None],
        }
    )

    result = df.select(
        url_query_encode(pl.struct("name", "page", "active", "missing")).alias("query")
    )["query"]

    assert result.to_list() == [
        "name=Ada+Lovelace&page=2&active=True&missing=None",
        "name=Grace+%26+Hopper&page=3&active=False&missing=None",
    ]


def test_url_query_encode_doseq() -> None:
    df = pl.DataFrame({"tags": [["python", "polars"], []]})

    encoded = df.select(url_query_encode(pl.struct("tags"), doseq=True))["tags"]
    scalar = df.select(url_query_encode(pl.struct("tags"), doseq=False))["tags"]

    assert encoded.to_list() == ["tags=python&tags=polars", ""]
    assert scalar.to_list() == ["tags=%5B%27python%27%2C+%27polars%27%5D", "tags=%5B%5D"]


def test_url_query_encode_list_struct_preserves_duplicates() -> None:
    df = pl.DataFrame({"json": ['{"tag":"a b","page":2,"missing":null}']})
    items = pl.col("json").str.json_object_items()

    result = df.select(url_query_encode(items).alias("query"))["query"]

    assert result.to_list() == ["tag=a+b&page=2&missing=None"]


def test_url_query_encode_null_input() -> None:
    df = pl.DataFrame(
        {"query": [None]},
        schema={"query": pl.Struct({"value": pl.String})},
    )

    assert df.select(url_query_encode("query"))["query"].to_list() == [None]


def test_url_query_encode_validates_doseq() -> None:
    with pytest.raises(TypeError, match="doseq must be a bool"):
        url_query_encode(pl.struct("value"), doseq="yes")  # type: ignore[arg-type]


def test_url_query_encode_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_url_query_encode, "_native_library_available", lambda: False)
    df = pl.DataFrame({"name": ["Ada Lovelace"], "tags": [["a", "b"]]})

    result = df.select(url_query_encode(pl.struct("name", "tags"), doseq=True).alias("query"))

    assert result["query"].to_list() == ["name=Ada+Lovelace&tags=a&tags=b"]

from __future__ import annotations

import polars as pl
import pytest
from polars_list_math import _url_build, url_build


def test_url_build_matches_urlunparse() -> None:
    df = pl.DataFrame(
        {
            "scheme": ["https", "http"],
            "host": ["example.com", "localhost:8080"],
            "path": ["/search", "api/items"],
            "params": ["", "v=1"],
            "query": ["q=polars", "page=2"],
            "fragment": ["results", ""],
        }
    )

    result = df.select(
        url_build(
            scheme="scheme",
            netloc="host",
            path="path",
            params="params",
            query="query",
            fragment="fragment",
        ).alias("url")
    )["url"]

    assert result.to_list() == [
        "https://example.com/search?q=polars#results",
        "http://localhost:8080/api/items;v=1?page=2",
    ]


def test_url_build_optional_and_literal_components() -> None:
    df = pl.DataFrame({"path": ["docs", None]})

    result = df.select(
        url_build(
            scheme=pl.lit("https"),
            netloc=pl.lit("example.com"),
            path="path",
        ).alias("url")
    )["url"]

    assert result.to_list() == ["https://example.com/docs", "https://example.com"]
    assert pl.select(url_build()).item() == ""


def test_url_build_does_not_encode_components() -> None:
    result = pl.select(
        url_build(
            path=pl.lit("/search results"),
            query=pl.lit("q=a b"),
        )
    ).item()

    assert result == "/search results?q=a b"


def test_url_build_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_url_build, "_native_library_available", lambda: False)

    result = pl.select(
        url_build(
            scheme=pl.lit("https"),
            netloc=pl.lit("example.com"),
            path=pl.lit("/docs"),
        ).alias("url")
    )

    assert result["url"].to_list() == ["https://example.com/docs"]

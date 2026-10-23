from __future__ import annotations

import polars as pl
import polars_list_math  # noqa: F401
import pytest
from polars_list_math import _list_zip


def test_list_zip_matches_issue_example() -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 2], [3], None, [None]],
            "b": [[10, 20], [30, 35], [40], [35]],
        }
    )

    out = df.with_columns(pl.col("a").list.zip(pl.col("b")).alias("zipped"))

    assert out["zipped"].to_list() == [
        [
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
        ],
        [{"a": 3, "b": 30}],
        None,
        [{"a": None, "b": 35}],
    ]


def test_zip_list_accepts_more_than_two_columns_and_custom_fields() -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 2], [3]],
            "b": [[10, 20], [30, 35]],
            "c": [["x", "y"], ["z", "ignored"]],
        }
    )

    out = df.with_columns(pl.col("a").list.zip("b", "c", fields=["a", "b", "c"]).alias("zipped"))

    assert out["zipped"].to_list() == [
        [{"a": 1, "b": 10, "c": "x"}, {"a": 2, "b": 20, "c": "y"}],
        [{"a": 3, "b": 30, "c": "z"}],
    ]


def test_pad_true_matches_longest_list() -> None:
    df = pl.DataFrame(
        {
            "a": [[1], [], None],
            "b": [[10, 20], [30], [40]],
        }
    )

    out = df.with_columns(pl.col("a").list.zip(pl.col("b"), pad=True).alias("zipped"))

    assert out["zipped"].to_list() == [
        [{"a": 1, "b": 10}, {"a": None, "b": 20}],
        [{"a": None, "b": 30}],
        None,
    ]


def test_lazy_frame_usage() -> None:
    df = pl.DataFrame({"a": [[1, 2]], "b": [["x", "y"]]})

    out = df.lazy().with_columns(pl.col("a").list.zip("b").alias("z")).collect()

    assert out["z"].to_list() == [[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]]


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="at least two"):
        pl.col("a").list.zip()

    with pytest.raises(ValueError, match="exactly 2"):
        pl.col("a").list.zip("b", fields=["a"])

    with pytest.raises(ValueError, match="unique"):
        pl.col("a").list.zip("b", fields=["same", "same"])

    with pytest.raises(TypeError, match="single string"):
        pl.col("a").list.zip("b", fields="ab")

    with pytest.raises(TypeError, match="bool"):
        pl.col("a").list.zip("b", pad="yes")  # type: ignore[arg-type]


def test_python_fallback_when_native_library_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_list_zip, "_native_library_available", lambda: False)
    df = pl.DataFrame({"a": [[1], []], "b": [[10, 20], [30]]})

    out = df.with_columns(pl.col("a").list.zip("b", pad=True).alias("zipped"))

    assert out["zipped"].to_list() == [
        [{"a": 1, "b": 10}, {"a": None, "b": 20}],
        [{"a": None, "b": 30}],
    ]


@pytest.mark.skipif(
    not _list_zip._native_library_available(),
    reason="native plugin has not been built",
)
def test_native_plugin_errors() -> None:
    df = pl.DataFrame({"a": [[1, 2]], "not_list": [1]})

    with pytest.raises(pl.exceptions.ComputeError, match="has more than one occurrence"):
        df.select(pl.col("a").list.zip(pl.col("a")))

    with pytest.raises(pl.exceptions.ComputeError, match="expected `List`"):
        df.select(pl.col("a").list.zip(pl.col("not_list")))


@pytest.fixture(params=["native", "fallback"])
def zip_backend(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.param == "native":
        if not _list_zip._native_library_available():
            pytest.skip("native plugin has not been built")
    else:
        monkeypatch.setattr(_list_zip, "_native_library_available", lambda: False)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("pad", [False, True])
def test_broadcast_lengths_nulls_and_empty_lists(zip_backend: None, lazy: bool, pad: bool) -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 2], [], None, [None], [3]],
            "b": [[10], [20], [30], [40, 50], []],
            "s": ["x", "y", "z", None, "w"],
        }
    )
    expr = _list_zip.list_zip("s", "a", "b", broadcast=True, pad=pad).alias("z")
    result = df.lazy().select(expr).collect() if lazy else df.select(expr)
    expected = [
        [{"s": "x", "a": 1, "b": 10}],
        [],
        None,
        [{"s": None, "a": None, "b": 40}],
        [],
    ]
    if pad:
        expected[0].append({"s": "x", "a": 2, "b": None})
        expected[1] = [{"s": "y", "a": None, "b": 20}]
        expected[3].append({"s": None, "a": None, "b": 50})
        expected[4] = [{"s": "w", "a": 3, "b": None}]
    assert result["z"].to_list() == expected


def test_broadcast_literals_and_namespace(zip_backend: None) -> None:
    df = pl.DataFrame({"a": [[1, 2], [3], []]})
    out = df.select(
        pl.col("a").list.zip(
            pl.lit("RUB"),
            pl.lit(None),
            pl.lit(True),
            broadcast=True,
            fields=["item", "currency", "missing", "flag"],
        )
    )
    assert out.to_series().to_list() == [
        [{"item": i, "currency": "RUB", "missing": None, "flag": True} for i in [1, 2]],
        [{"item": 3, "currency": "RUB", "missing": None, "flag": True}],
        [],
    ]


@pytest.mark.parametrize("empty", [False, True])
def test_broadcast_preserves_schema_and_nested_values(zip_backend: None, empty: bool) -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series([[[1], [2]], []], dtype=pl.List(pl.List(pl.Int16))),
            "s": pl.Series([{"x": 1}, None], dtype=pl.Struct({"x": pl.Int32})),
        }
    )
    if empty:
        df = df.head(0)
    expr = _list_zip.list_zip("a", "s", broadcast=True).alias("z")
    out = df.lazy().select(expr).collect()
    assert out.schema["z"] == pl.List(
        pl.Struct(
            {
                "a": pl.List(pl.Int16),
                "s": pl.Struct({"x": pl.Int32}),
            }
        )
    )
    assert out["z"].to_list() == (
        []
        if empty
        else [
            [{"a": [1], "s": {"x": 1}}, {"a": [2], "s": {"x": 1}}],
            [],
        ]
    )


@pytest.mark.parametrize("empty", [False, True])
def test_list_zip_preserves_schema_without_broadcast(zip_backend: None, empty: bool) -> None:
    inner = pl.Struct({"n": pl.Int32})
    df = pl.DataFrame(
        {"a": [[1, 2], [], None], "b": [[{"n": 7}], [], [{"n": 9}]]},
        schema={"a": pl.List(pl.Int16), "b": pl.List(inner)},
    )
    if empty:
        df = df.head(0)
    query = df.lazy().select(_list_zip.list_zip("a", "b").alias("z"))
    expected_dtype = pl.List(pl.Struct({"a": pl.Int16, "b": inner}))
    assert query.collect_schema() == pl.Schema({"z": expected_dtype})
    result = query.collect()
    assert result.schema == pl.Schema({"z": expected_dtype})
    assert result["z"].to_list() == ([] if empty else [[{"a": 1, "b": {"n": 7}}], [], None])


def test_broadcast_null_list_and_empty_frame_with_literal(zip_backend: None) -> None:
    df = pl.DataFrame({"a": pl.Series([None], dtype=pl.List(pl.Int64))})
    expr = _list_zip.list_zip("a", pl.lit(5), broadcast=True, fields=["a", "b"])
    assert df.select(expr).to_series().to_list() == [None]
    assert df.head(0).select(expr).height == 0


def test_broadcast_requires_list_and_explicit_opt_in(zip_backend: None) -> None:
    df = pl.DataFrame({"a": [[1, 2]], "s": [3]})
    with pytest.raises((TypeError, pl.exceptions.PolarsError)):
        df.select(_list_zip.list_zip("a", "s"))
    with pytest.raises((TypeError, pl.exceptions.PolarsError)):
        df.select(_list_zip.list_zip("s", pl.lit(4), broadcast=True))
    with pytest.raises(TypeError, match="broadcast must be a bool"):
        _list_zip.list_zip("a", "s", broadcast=1)  # type: ignore[arg-type]


def test_broadcast_list_only_and_chunked_input(zip_backend: None) -> None:
    df = pl.concat(
        [
            pl.DataFrame({"a": [[1, 2]], "b": [[3]], "s": [4]}),
            pl.DataFrame({"a": [[5]], "b": [[6, 7]], "s": [8]}),
        ],
        rechunk=False,
    )
    for pad in [False, True]:
        actual = df.select(_list_zip.list_zip("a", "b", broadcast=True, pad=pad))
        expected = df.select(_list_zip.list_zip("a", "b", pad=pad))
        assert actual.to_series().to_list() == expected.to_series().to_list()
    assert df.select(_list_zip.list_zip("a", "s", broadcast=True)).to_series().to_list() == [
        [{"a": 1, "s": 4}, {"a": 2, "s": 4}],
        [{"a": 5, "s": 8}],
    ]


@pytest.mark.parametrize("pad", [False, True])
def test_broadcast_matches_explicit_repetition(zip_backend: None, pad: bool) -> None:
    df = pl.DataFrame(
        {
            "a": [list(range(i % 7)) for i in range(50)],
            "b": [list(range(i % 5)) for i in range(50)],
            "s": [None if i % 3 == 0 else i / 2 for i in range(50)],
        }
    )
    length_fn = pl.max_horizontal if pad else pl.min_horizontal
    length = length_fn(pl.col("a").list.len(), pl.col("b").list.len())
    actual = (
        df.lazy()
        .select(_list_zip.list_zip("a", "b", "s", broadcast=True, pad=pad).alias("z"))
        .collect(engine="streaming")
    )
    expected = df.select(
        _list_zip.list_zip(
            "a", "b", pl.col("s").repeat_by(length), pad=pad, fields=["a", "b", "s"]
        ).alias("z")
    )
    assert actual.equals(expected)

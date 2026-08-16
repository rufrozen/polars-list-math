"""Tests for typed Polars projections."""

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


class Item(tp.Schema):
    value = tp.Field[str]()
    score = tp.Field[int]()


class Payload(tp.Schema):
    title = tp.Field[str]()
    items = tp.ListStruct[Item](flat_alias="entries")


class Row(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    payload = tp.Struct[Payload](flat_alias="p")
    ignored = tp.Field[bool]()


class RowView(tp.View):
    identifier = tp.ViewField[str](Row.request_id)
    title = tp.ViewField[str](Row.payload.fields.title)
    values = tp.ViewField[list[str]](Row.payload.fields.items.item.value)
    scores = tp.ViewField[list[int]](
        Row.payload.fields.items.item.score,
        alias="item_scores",
    )


@pytest.fixture
def frame() -> pl.DataFrame:
    rows = [
        Row(
            request_id="one",
            payload=Payload(
                title="First",
                items=[Item(value="a", score=1), Item(value="b", score=2)],
            ),
            ignored=True,
        ),
        Row(
            request_id="two",
            payload=Payload(title="Second", items=[]),
            ignored=False,
        ),
    ]
    return Row.to_frame_many(rows)


def test_select_builds_dataframe_projection(frame: pl.DataFrame) -> None:
    selected = RowView.select(frame)

    assert isinstance(selected, pl.DataFrame)
    assert selected.to_dict(as_series=False) == {
        "identifier": ["one", "two"],
        "title": ["First", "Second"],
        "values": [["a", "b"], []],
        "item_scores": [[1, 2], []],
    }


def test_select_preserves_lazy_execution(frame: pl.DataFrame) -> None:
    selected = RowView.select(frame.lazy())

    assert isinstance(selected, pl.LazyFrame)
    assert selected.collect().columns == [
        "identifier",
        "title",
        "values",
        "item_scores",
    ]


def test_select_automatically_supports_flat_dataframe_and_lazyframe(
    frame: pl.DataFrame,
) -> None:
    rows = list(Row.iter_frame(frame))
    flat = Row.to_flat_frame_many(rows)

    assert flat.columns == [
        "requestId",
        "p:title",
        "p:entries:value",
        "p:entries:score",
        "ignored",
    ]
    nested_selected = RowView.select(frame)
    selected = RowView.select(flat)
    lazy_selected = RowView.select(flat.lazy())

    assert isinstance(nested_selected, pl.DataFrame)
    expected = nested_selected.to_dict(as_series=False)
    assert isinstance(selected, pl.DataFrame)
    assert selected.to_dict(as_series=False) == expected
    assert isinstance(lazy_selected, pl.LazyFrame)
    assert lazy_selected.collect().to_dict(as_series=False) == expected


def test_from_frame_returns_typed_view_objects(frame: pl.DataFrame) -> None:
    views = RowView.from_frame(frame)

    assert views == [
        RowView(identifier="one", title="First", values=["a", "b"], scores=[1, 2]),
        RowView(identifier="two", title="Second", values=[], scores=[]),
    ]
    assert views[0].identifier == "one"
    assert views[0].scores == [1, 2]
    assert views[0].to_dict(by_alias=True)["item_scores"] == [1, 2]


def test_view_exposes_source_metadata() -> None:
    assert RowView.identifier.source is Row.request_id
    assert RowView.values.source.polars_path == ("payload", "items", "[]", "value")
    assert RowView.values.source.flat_path == ("p", "entries", "value")
    assert RowView.values.source.flat_name == "p:entries:value"
    assert tuple(RowView.model_fields()) == ("identifier", "title", "values", "scores")


def test_columns_expose_explicit_nested_and_flat_expressions(
    frame: pl.DataFrame,
) -> None:
    rows = list(Row.iter_frame(frame))
    flat = Row.to_flat_frame_many(rows)
    source = Row.payload.fields.items.item.value

    assert frame.select(source.nested_expr()).to_series().to_list() == [
        ["a", "b"],
        [],
    ]
    assert flat.select(source.flat_expr()).to_series().to_list() == [
        ["a", "b"],
        [],
    ]
    assert not hasattr(source, "expr")
    with pytest.raises(TypeError, match="not callable"):
        source()  # type: ignore[operator]


def test_missing_root_column_becomes_null_without_losing_rows(
    frame: pl.DataFrame,
) -> None:
    selected = RowView.select(frame.drop("requestId"))

    assert isinstance(selected, pl.DataFrame)
    assert selected["identifier"].to_list() == [None, None]
    assert selected["title"].to_list() == ["First", "Second"]


def test_missing_nested_fields_become_typed_null_columns() -> None:
    frame = pl.DataFrame(
        {
            "requestId": ["one", "two"],
            "payload": [{"title": "First"}, {"title": "Second"}],
        },
        schema={
            "requestId": pl.String,
            "payload": pl.Struct({"title": pl.String}),
        },
    )

    selected = RowView.select(frame)

    assert isinstance(selected, pl.DataFrame)
    assert selected.to_dict(as_series=False) == {
        "identifier": ["one", "two"],
        "title": ["First", "Second"],
        "values": [None, None],
        "item_scores": [None, None],
    }
    assert selected.schema["values"] == pl.List(pl.String)
    assert selected.schema["item_scores"] == pl.List(pl.Int64)


def test_missing_columns_are_supported_for_lazy_frames(frame: pl.DataFrame) -> None:
    selected = RowView.select(frame.drop("requestId", "payload").lazy())

    assert isinstance(selected, pl.LazyFrame)
    assert selected.collect().height == frame.height


def test_view_rejects_invalid_declarations_and_constructor_values() -> None:
    with pytest.raises(TypeError, match="source must be a Column"):
        tp.ViewField("requestId")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="duplicate view alias"):

        class DuplicateView(tp.View):
            first = tp.ViewField[str](Row.request_id, alias="same")
            second = tp.ViewField[str](Row.request_id, alias="same")

    with pytest.raises(TypeError, match="Missing required field"):
        RowView()

    with pytest.raises(TypeError, match="Unexpected field"):
        RowView(identifier="one", title="First", values=[], scores=[], extra=True)

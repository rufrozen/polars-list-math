"""Tests for typed Polars models and schemas."""

from datetime import datetime, timedelta

import polars as pl
import polars_list_math.typed_polars as tp
import pytest


class Suggestion(tp.Schema):
    value = tp.Field[str]()
    corrected_query = tp.Field[str](alias="correctedQuery")


class Completion(tp.Schema):
    prefix = tp.Field[str](alias="queryPrefix")
    suggestions = tp.ListStruct[Suggestion]()


class Row(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    position = tp.Field[tp.I32]()
    score = tp.Field[tp.F32 | None](default=None)
    completion = tp.Struct[Completion](alias="completionData")
    tags = tp.Field[list[str]](default_factory=list)
    created_at = tp.Field[tp.TimestampMs]()
    elapsed = tp.Field[tp.DurationMs]()


def make_row(*, request_id: str = "request-1") -> Row:
    return Row(
        request_id=request_id,
        position=7,
        completion=Completion(
            prefix="по",
            suggestions=[
                Suggestion(value="поле", corrected_query="поле"),
                Suggestion(value="полёт", corrected_query="полет"),
            ],
        ),
        # TimestampMs intentionally models a timezone-naive Polars datetime.
        created_at=datetime(2025, 1, 2, 3, 4, 5, 678000),  # noqa: DTZ001
        elapsed=timedelta(milliseconds=12),
    )


def test_builds_exact_polars_schema() -> None:
    assert Row.schema == pl.Schema(
        {
            "requestId": pl.String,
            "position": pl.Int32,
            "score": pl.Float32,
            "completionData": pl.Struct(
                {
                    "queryPrefix": pl.String,
                    "suggestions": pl.List(
                        pl.Struct(
                            {
                                "value": pl.String,
                                "correctedQuery": pl.String,
                            }
                        )
                    ),
                }
            ),
            "tags": pl.List(pl.String),
            "created_at": pl.Datetime("ms"),
            "elapsed": pl.Duration("ms"),
        }
    )
    assert Row.polars_schema() is Row.polars_schema()


def test_serializes_and_deserializes_nested_values_with_aliases() -> None:
    row = make_row()
    serialized = row.to_dict(by_alias=True)

    assert serialized["requestId"] == "request-1"
    assert serialized["completionData"] == {
        "queryPrefix": "по",
        "suggestions": [
            {"value": "поле", "correctedQuery": "поле"},
            {"value": "полёт", "correctedQuery": "полет"},
        ],
    }
    assert Row.from_dict(serialized, by_alias=True) == row


def test_default_factory_creates_independent_values() -> None:
    first = make_row(request_id="first")
    second = make_row(request_id="second")

    first.tags.append("changed")

    assert second.tags == []


def test_round_trips_one_or_many_rows_through_dataframe() -> None:
    first = make_row(request_id="first")
    second = make_row(request_id="second")

    frame = Row.to_frame_many([first, second])

    assert frame.schema == Row.schema
    assert Row.from_frame(frame, strict_schema=True) == first
    assert list(Row.iter_frame(frame, strict_schema=True)) == [first, second]


def test_to_frame_rejects_null_for_non_nullable_field_in_strict_mode() -> None:
    class StrictRow(tp.Schema):
        value = tp.Field[str]()

    with pytest.raises(TypeError, match=r"Field 'value' does not accept None"):
        StrictRow(value=None).to_frame()  # type: ignore[arg-type]


def test_to_frame_uses_declared_default_for_null_in_non_strict_mode() -> None:
    class DefaultRow(tp.Schema):
        value = tp.Field[str](default="fallback")

    frame = DefaultRow(value=None).to_frame(strict=False)  # type: ignore[arg-type]

    assert frame.to_dict(as_series=False) == {"value": ["fallback"]}


def test_to_frame_keeps_explicitly_nullable_field_null() -> None:
    class NullableRow(tp.Schema):
        value = tp.Field[str | None](default=None)

    assert NullableRow().to_frame().to_dict(as_series=False) == {"value": [None]}


def test_non_strict_null_without_default_is_rejected() -> None:
    class RequiredRow(tp.Schema):
        value = tp.Field[str]()

    with pytest.raises(TypeError, match=r"Field 'value'.*has no default"):
        RequiredRow(value=None).to_frame(strict=False)  # type: ignore[arg-type]


def test_flat_structs_are_part_of_the_fixed_schema() -> None:
    class FlatRow(tp.Schema):
        completion = tp.Struct[Completion](alias="completionData", flat=True)

    frame = FlatRow(completion=make_row().completion).to_frame()

    assert FlatRow.schema == frame.schema
    assert frame.schema == pl.Schema(
        {
            "completionData:queryPrefix": pl.String,
            "completionData:suggestions": pl.List(
                pl.Struct({"value": pl.String, "correctedQuery": pl.String})
            ),
        }
    )
    assert frame.to_dict(as_series=False) == {
        "completionData:queryPrefix": ["по"],
        "completionData:suggestions": [
            [
                {"value": "поле", "correctedQuery": "поле"},
                {"value": "полёт", "correctedQuery": "полет"},
            ]
        ],
    }
    assert FlatRow.from_frame(frame, strict_schema=True).completion.prefix == "по"


def test_flat_frame_preserves_nested_list_struct_boundaries() -> None:
    class Child(tp.Schema):
        value = tp.Field[int]()

    class Parent(tp.Schema):
        children = tp.ListStruct[Child](alias="kids", flat=True, flat_divider="/")

    class NestedRow(tp.Schema):
        parents = tp.ListStruct[Parent](alias="parents", flat=True)

    frame = NestedRow(
        parents=[
            Parent(children=[Child(value=1), Child(value=2)]),
            Parent(children=[Child(value=3)]),
        ]
    ).to_frame()

    assert frame.schema == pl.Schema({"parents:kids/value": pl.List(pl.List(pl.Int64))})
    assert frame["parents:kids/value"].to_list() == [[[1, 2], [3]]]
    assert NestedRow.from_frame(frame).parents[1].children[0].value == 3


def test_struct_alias_and_custom_divider_define_flat_namespace() -> None:
    class Payload(tp.Schema):
        title = tp.Field[str](alias="itemTitle")

    class AliasedRow(tp.Schema):
        payload = tp.Struct[Payload](alias="p", flat=True, flat_divider=".")

    row = AliasedRow(payload=Payload(title="result"))

    assert AliasedRow.schema == pl.Schema({"p.itemTitle": pl.String})
    assert row.to_frame().to_dict(as_series=False) == {"p.itemTitle": ["result"]}

    with pytest.raises(TypeError, match="non-empty"):
        tp.Struct[Payload](flat=True, flat_divider="")


def test_hybrid_paths_work_across_flat_and_nested_boundaries() -> None:
    class Details(tp.Schema):
        code = tp.Field[int]()

    class Item(tp.Schema):
        details = tp.Struct[Details]()

    class HybridRow(tp.Schema):
        items = tp.ListStruct[Item](flat=True, flat_divider=".")

    frame = HybridRow(items=[Item(details=Details(code=7))]).to_frame()

    assert frame.schema == pl.Schema({"items.details": pl.List(pl.Struct({"code": pl.Int64}))})
    assert frame.select(HybridRow.items.item.details.fields.code.expr()).to_series().to_list() == [
        [7]
    ]
    assert HybridRow.from_frame(frame).items[0].details.code == 7


def test_builds_empty_typed_frame_from_one_shot_iterable() -> None:
    rows = (row for row in [])

    frame = Row.to_frame_many(rows)

    assert frame.height == 0
    assert frame.schema == Row.schema


def test_indexed_fields_support_single_schema_inheritance() -> None:
    class Base(tp.Schema):
        inherited = tp.Field[str]()

    class Child(Base):
        own = tp.Field[int]()

    base = Base(inherited="base")
    child = Child(inherited="child", own=2)

    assert base.inherited == "base"
    assert child.inherited == "child"
    assert child.own == 2
    assert child.to_dict() == {"inherited": "child", "own": 2}


def test_rejects_multiple_inheritance_from_field_schemas() -> None:
    class Left(tp.Schema):
        left = tp.Field[str]()

    class Right(tp.Schema):
        right = tp.Field[str]()

    with pytest.raises(TypeError, match="multiple inheritance"):

        class Combined(Left, Right):
            pass


def test_nested_columns_build_paths_and_work_in_select() -> None:
    frame = make_row().to_frame()
    prefix = Row.completion.fields.prefix
    values = Row.completion.fields.suggestions.item.value

    assert prefix.python_path == ("completion", "prefix")
    assert prefix.polars_path == ("completionData", "queryPrefix")
    assert values.python_path == ("completion", "suggestions", "item", "value")
    assert values.polars_path == ("completionData", "suggestions", "[]", "value")
    assert frame.select(prefix.expr(), values.expr()).to_dict(as_series=False) == {
        "queryPrefix": ["по"],
        "suggestions": [["поле", "полёт"]],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unknown": 1}, "Unexpected field"),
        ({}, "Missing required field"),
    ],
)
def test_constructor_rejects_invalid_keys(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(TypeError, match=message):
        Row(**kwargs)


def test_from_dict_rejects_missing_and_extra_keys() -> None:
    with pytest.raises(TypeError, match="Missing required key"):
        Row.from_dict({})

    data = make_row().to_dict()
    with pytest.raises(TypeError, match="Unexpected key"):
        Row.from_dict({**data, "extra": True})


def test_schema_declaration_rejects_duplicate_aliases() -> None:
    with pytest.raises(TypeError, match="duplicate Polars alias"):

        class DuplicateAliases(tp.Schema):
            first = tp.Field[str](alias="same")
            second = tp.Field[str](alias="same")


def test_schema_declaration_requires_generic_type() -> None:
    with pytest.raises(TypeError, match="must have a generic type"):

        class MissingType(tp.Schema):
            value = tp.Field()


def test_assert_frame_schema_can_allow_extra_columns() -> None:
    frame = make_row().to_frame().with_columns(pl.lit(True).alias("extra"))

    with pytest.raises(TypeError, match="Unexpected DataFrame schema"):
        Row.assert_frame_schema(frame)
    Row.assert_frame_schema(frame, allow_extra=True)


class FlexiblePayload(tp.Schema):
    title = tp.Field[str]()
    extras = tp.Extras()


class FlexibleRow(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    payload = tp.Struct[FlexiblePayload]()
    extras = tp.Extras()


def test_extras_capture_unknown_columns_and_nested_fields() -> None:
    row = FlexibleRow.from_dict(
        {
            "requestId": "one",
            "rank": 3,
            "payload": {"title": "Result", "debug": True},
        },
        by_alias=True,
    )

    assert row.extras == {"rank": 3}
    assert row.payload.extras == {"debug": True}
    assert row.to_dict(by_alias=True) == {
        "requestId": "one",
        "payload": {"title": "Result", "debug": True},
        "rank": 3,
    }


def test_extras_are_inferred_as_separate_polars_columns_and_struct_fields() -> None:
    rows = [
        FlexibleRow(
            request_id="one",
            payload=FlexiblePayload(title="First", extras={"debug": True}),
            extras={"rank": 1},
        ),
        FlexibleRow(
            request_id="two",
            payload=FlexiblePayload(title="Second", extras={"source": "web"}),
            extras={"note": "new"},
        ),
    ]

    frame = FlexibleRow.to_frame_many(rows)

    assert frame.columns == ["requestId", "payload", "rank", "note"]
    assert frame.schema["rank"] == pl.Int64
    assert frame.schema["note"] == pl.String
    assert frame.schema["payload"] == pl.Struct(
        {"title": pl.String, "debug": pl.Boolean, "source": pl.String}
    )
    restored = FlexibleRow.from_frame(frame, index=1)
    assert restored.extras == {"rank": None, "note": "new"}
    assert restored.payload.extras == {"debug": None, "source": "web"}


def test_explicit_extra_schema_supports_null_values_and_nested_extras() -> None:
    row = FlexibleRow(
        request_id="one",
        payload=FlexiblePayload(title="First", extras={"debug": None}),
        extras={"rank": None},
    )
    extra_schema = {
        "rank": pl.Int32,
        "payload": {"debug": pl.Boolean},
    }

    frame = row.to_frame(extra_schema=extra_schema)

    assert frame.schema == pl.Schema(
        {
            "requestId": pl.String,
            "payload": pl.Struct({"title": pl.String, "debug": pl.Boolean}),
            "rank": pl.Int32,
        }
    )


def test_extras_use_column_builder_without_from_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        FlexibleRow(
            request_id="one",
            payload=FlexiblePayload(title="First", extras={"debug": True}),
            extras={"rank": 1},
        )
    ]

    def reject_from_dicts(*args: object, **kwargs: object) -> None:
        raise AssertionError("to_frame_many must not call pl.from_dicts")

    monkeypatch.setattr(pl, "from_dicts", reject_from_dicts)

    frame = FlexibleRow.to_frame_many(rows)

    assert frame.to_dicts() == [
        {
            "requestId": "one",
            "payload": {"title": "First", "debug": True},
            "rank": 1,
        }
    ]


def test_list_struct_extras_are_resolved_across_all_items() -> None:
    class Item(tp.Schema):
        value = tp.Field[str]()
        extras = tp.Extras()

    class ItemRow(tp.Schema):
        items = tp.ListStruct[Item]()

    frame = ItemRow.to_frame_many(
        [
            ItemRow(items=[Item(value="a", extras={"rank": 1})]),
            ItemRow(items=[Item(value="b", extras={"source": "web"})]),
        ]
    )

    assert frame.schema["items"] == pl.List(
        pl.Struct({"value": pl.String, "rank": pl.Int64, "source": pl.String})
    )
    assert frame.to_dicts() == [
        {"items": [{"value": "a", "rank": 1, "source": None}]},
        {"items": [{"value": "b", "rank": None, "source": "web"}]},
    ]


def test_extras_reject_conflicts_and_invalid_declarations() -> None:
    row = FlexibleRow(
        request_id="one",
        payload=FlexiblePayload(title="First"),
        extras={"requestId": "conflict"},
    )
    with pytest.raises(TypeError, match="Extras conflict"):
        row.to_dict(by_alias=True)

    with pytest.raises(TypeError, match="only one Extras"):

        class TooManyExtras(tp.Schema):
            first = tp.Extras()
            second = tp.Extras()

    with pytest.raises(TypeError, match="does not declare Extras"):
        Row.polars_schema({"dynamic": pl.String})


class ProjectedSuggestion(tp.Schema):
    value = tp.Include(Suggestion.value)


class ProjectedCompletion(tp.Schema):
    prefix = tp.Include(Completion.prefix)
    suggestions = tp.IncludeListStruct(Completion.suggestions, ProjectedSuggestion)


class ProjectedRow(tp.Schema):
    request_id = tp.Include(Row.request_id)
    score = tp.Include(Row.score)
    completion = tp.IncludeStruct(Row.completion, ProjectedCompletion)
    tags = tp.Include(Row.tags)


def test_include_descriptors_build_a_regular_partial_schema() -> None:
    assert ProjectedRow.schema == pl.Schema(
        {
            "requestId": pl.String,
            "score": pl.Float32,
            "completionData": pl.Struct(
                {
                    "queryPrefix": pl.String,
                    "suggestions": pl.List(pl.Struct({"value": pl.String})),
                }
            ),
            "tags": pl.List(pl.String),
        }
    )

    row = ProjectedRow(
        request_id="one",
        completion=ProjectedCompletion(
            prefix="по",
            suggestions=[ProjectedSuggestion(value="поле")],
        ),
    )

    assert row.score is None
    assert row.tags == []
    assert row.to_frame().to_dict(as_series=False) == {
        "requestId": ["one"],
        "score": [None],
        "completionData": [{"queryPrefix": "по", "suggestions": [{"value": "поле"}]}],
        "tags": [[]],
    }


def test_include_can_flatten_a_nested_scalar_field() -> None:
    class PrefixOnly(tp.Schema):
        prefix = tp.Include(Row.completion.fields.prefix)

    assert PrefixOnly.schema == pl.Schema({"queryPrefix": pl.String})
    assert PrefixOnly(prefix="по").to_frame()["queryPrefix"].to_list() == ["по"]


def test_include_rejects_incompatible_source_kinds() -> None:
    with pytest.raises(TypeError, match="IncludeStruct"):
        tp.Include(Row.completion)

    with pytest.raises(TypeError, match="StructColumn"):
        tp.IncludeStruct(Row.request_id, ProjectedCompletion)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="ListStructColumn"):
        tp.IncludeListStruct(Row.request_id, ProjectedSuggestion)  # type: ignore[arg-type]


class DictionaryRow(tp.Schema):
    counters = tp.Field[dict[str, int]](default_factory=dict)
    suggestions = tp.Field[dict[str, Suggestion]]()


def test_dict_fields_use_list_of_key_value_structs() -> None:
    assert DictionaryRow.schema == pl.Schema(
        {
            "counters": pl.List(pl.Struct({"key": pl.String, "value": pl.Int64})),
            "suggestions": pl.List(
                pl.Struct(
                    {
                        "key": pl.String,
                        "value": pl.Struct(
                            {
                                "value": pl.String,
                                "correctedQuery": pl.String,
                            }
                        ),
                    }
                )
            ),
        }
    )


def test_dict_fields_round_trip_through_dict_and_dataframe() -> None:
    row = DictionaryRow(
        counters={"seen": 3, "clicked": 1},
        suggestions={
            "first": Suggestion(value="поле", corrected_query="поле"),
        },
    )

    serialized = row.to_dict(by_alias=True)

    assert serialized == {
        "counters": [
            {"key": "seen", "value": 3},
            {"key": "clicked", "value": 1},
        ],
        "suggestions": [
            {
                "key": "first",
                "value": {"value": "поле", "correctedQuery": "поле"},
            }
        ],
    }
    assert DictionaryRow.from_dict(serialized, by_alias=True) == row
    assert DictionaryRow.from_frame(row.to_frame(), strict_schema=True) == row


def test_from_dict_accepts_python_mapping_for_dict_field() -> None:
    row = DictionaryRow.from_dict(
        {
            "counters": {"seen": 3},
            "suggestions": {"first": {"value": "поле", "corrected_query": "поле"}},
        }
    )

    assert row.counters == {"seen": 3}
    assert row.suggestions["first"] == Suggestion(
        value="поле",
        corrected_query="поле",
    )


def test_dict_field_rejects_malformed_entries() -> None:
    with pytest.raises(TypeError, match="key.*value"):
        DictionaryRow.from_dict(
            {
                "counters": [{"key": "seen"}],
                "suggestions": [],
            }
        )

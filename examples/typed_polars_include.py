"""Build a smaller schema by including fields from a shared schema."""

import polars_list_math.typed_polars as tp


class Suggestion(tp.Schema):
    value = tp.Field[str]()
    score = tp.Field[float]()


class Completion(tp.Schema):
    prefix = tp.Field[str](alias="queryPrefix")
    suggestions = tp.ListStruct[Suggestion]()


class SearchRow(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    completion = tp.Struct[Completion](alias="completionData")


class CompactSuggestion(tp.Schema):
    value = tp.Include(Suggestion.value)


class CompactCompletion(tp.Schema):
    prefix = tp.Include(Completion.prefix)
    suggestions = tp.IncludeListStruct(Completion.suggestions, CompactSuggestion)


class CompactSearchRow(tp.Schema):
    request_id = tp.Include(SearchRow.request_id)
    completion = tp.IncludeStruct(SearchRow.completion, CompactCompletion)


row = CompactSearchRow(
    request_id="request-1",
    completion=CompactCompletion(
        prefix="pol",
        suggestions=[CompactSuggestion(value="polars")],
    ),
)

print(CompactSearchRow.schema)
print(row.to_frame())

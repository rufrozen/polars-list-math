"""Select a typed view from nested and flat DataFrames."""

import polars as pl
import polars_list_math.typed_polars as tp


class Item(tp.Schema):
    name = tp.Field[str]()
    score = tp.Field[float]()


class Result(tp.Schema):
    request_id = tp.Field[str](alias="requestId")
    items = tp.ListStruct[Item](flat_alias="matches")


class ResultView(tp.View):
    request_id = tp.ViewField[str](Result.request_id)
    names = tp.ViewField[list[str]](Result.items.item.name)
    scores = tp.ViewField[list[float]](
        Result.items.item.score,
        alias="item_scores",
    )


rows = [
    Result(
        request_id="request-1",
        items=[Item(name="polars", score=0.95), Item(name="python", score=0.8)],
    )
]
nested = Result.to_frame_many(rows)
flat = Result.to_flat_frame_many(rows)

print(ResultView.select(nested))
lazy_result = ResultView.select(flat.lazy())
assert isinstance(lazy_result, pl.LazyFrame)
print(lazy_result.collect())
print(ResultView.from_frame(nested))

assert isinstance(ResultView.select(nested), pl.DataFrame)

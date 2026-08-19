# Performance experiments

Изолированные эксперименты производительности Polars:

- [`dataframe_construction`](dataframe_construction) — построение DataFrame из
  `dataclass`, `dataclass(slots=True)` и `NamedTuple`.
- [`nested_struct_input`](nested_struct_input) — поддержка `dict`, tuple,
  `NamedTuple` и slots-dataclass внутри `Struct` и `List[Struct]`.
- [`typed_polars_construction`](typed_polars_construction) — полный путь
  построения из `typed_polars` против готовых вложенных `NamedTuple`.

Каждый эксперимент находится в отдельной папке и содержит собственный README
с описанием запуска и результатами контрольного прогона.

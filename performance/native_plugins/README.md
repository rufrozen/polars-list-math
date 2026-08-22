# Native plugin performance

Набор экспериментов сравнивает Rust expression plugins с их Python fallback на
одинаковых входных `DataFrame`. Каждая группа находится в отдельной папке:

- [`list_zip`](list_zip);
- [`list_combinations`](list_combinations) — обе combinations-функции;
- [`list_similarity`](list_similarity) — три similarity-функции;
- [`expected_value_of_game`](expected_value_of_game);
- [`json_object_items`](json_object_items);
- [`json_array_values`](json_array_values);
- [`url_query_encode`](url_query_encode);
- [`url_build`](url_build).

В каждой папке есть короткий baseline и дополнительный complex-сценарий с
длинными или глубоко вложенными данными. Перед измерением каждый план
прогревается. Для результата берётся лучшее время
из трёх повторов, а подготовка входных данных и построение expression не входят
в измеряемый участок. Тест также требует полного совпадения результата native и
fallback. Порог по времени намеренно отсутствует: он был бы нестабилен между
машинами и CI runners.

Сначала соберите актуальное нативное расширение, затем запустите эксперимент из
корня репозитория:

```bash
make develop
uv run pytest performance/native_plugins -v -s
```

Размер набора и число повторов настраиваются переменными окружения:

```bash
POLARS_LIST_MATH_PERF_ROWS=10000 \
POLARS_LIST_MATH_PERF_REPEATS=5 \
uv run pytest performance/native_plugins -v -s
```

Значения по умолчанию — 2 000 строк и 3 повтора. В выводе для каждой операции
показываются времена native/fallback и отношение `fallback / native`.

## Контрольный прогон

Прогон 2026-08-22 с Python 3.12.3 и Polars 1.42.1:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `list_zip` | 0.000394 s | 0.013344 s | 33.85x |
| `list_zip_complex` | 0.001124 s | 0.059398 s | 52.85x |
| `list_combinations` | 0.000498 s | 0.016221 s | 32.59x |
| `list_combinations_to` | 0.000860 s | 0.014249 s | 16.58x |
| `list_combinations_complex` | 0.006381 s | 0.122010 s | 19.12x |
| `list_combinations_to_complex` | 0.009351 s | 0.157177 s | 16.81x |
| `list_similarity` | 0.001043 s | 0.008305 s | 7.96x |
| `list_mean_similarity` | 0.002363 s | 0.043958 s | 18.60x |
| `list_mean_similarity_to` | 0.004872 s | 0.076319 s | 15.66x |
| `list_similarity_complex` | 0.010631 s | 0.048436 s | 4.56x |
| `list_mean_similarity_complex` | 0.038307 s | 0.580963 s | 15.17x |
| `list_mean_similarity_to_complex` | 0.060353 s | 0.940764 s | 15.59x |
| `expected_value_of_game` | 0.004624 s | 0.012254 s | 2.65x |
| `expected_value_of_game_complex` | 0.023965 s | 0.077370 s | 3.23x |
| `json_object_items` | 0.003640 s | 0.018851 s | 5.18x |
| `json_object_items_complex` | 0.022595 s | 0.053219 s | 2.36x |
| `json_array_values` | 0.001597 s | 0.012960 s | 8.12x |
| `json_array_values_complex` | 0.068937 s | 0.196068 s | 2.84x |
| `url_query_encode` | 0.000912 s | 0.006108 s | 6.70x |
| `url_query_encode_complex` | 0.006447 s | 0.029197 s | 4.53x |
| `url_build` | 0.001009 s | 0.005559 s | 5.51x |
| `url_build_complex` | 0.001187 s | 0.006419 s | 5.41x |

Это ориентир одного локального запуска, а не гарантированный performance
contract. Для сравнения изменений следует прогонять обе версии на одной машине.

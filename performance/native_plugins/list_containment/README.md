# Containment performance

Сравнение трёх Rust containment-функций с Python fallback. Все сценарии проверяют
совпадение результатов; подготовка данных и построение expressions не входят
в измерение. Используется лучший результат из трёх повторов после прогрева.

Baseline: короткие списки целых ID, повторы и null-элементы; для средних — четыре
целевых списка и три референса. Complex: списки длиной 32/28 для попарного
сравнения, восемь целевых списков длиной 12 и шесть референсов длиной 10 для
средних. Метрика направленная, порядок и повторы ID не влияют на результат.

```bash
uv run --no-sync --group dev maturin develop --release
uv run --no-sync --group dev pytest performance/native_plugins/list_containment -v -s
```

Размер и число повторов настраиваются через `POLARS_LIST_MATH_PERF_ROWS` и
`POLARS_LIST_MATH_PERF_REPEATS`. Debug-сборка для сравнения скорости не подходит.

## Контрольный прогон

2026-09-18, Linux x86_64, Python 3.12.3, Polars 1.42.1, Rust release,
2 000 строк, три повтора. Для изоляции от параллельной установки debug-сборки
плагин в этом прогоне загружен напрямую из `rust/target/release/lib_native.so`:

```bash
uv run --no-sync --group dev python - <<'PY'
from pathlib import Path
import pytest
from polars_list_math import _list_containment

_list_containment._LIB = Path('rust/target/release/lib_native.so').resolve()
raise SystemExit(pytest.main(['performance/native_plugins/list_containment', '-v', '-s']))
PY
```

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `list_containment` | 0.000877 s | 0.007259 s | 8.28x |
| `list_mean_containment` | 0.001711 s | 0.026294 s | 15.37x |
| `list_mean_containment_to` | 0.003287 s | 0.031218 s | 9.50x |
| `list_containment_complex` | 0.007834 s | 0.036052 s | 4.60x |
| `list_mean_containment_complex` | 0.023861 s | 0.113616 s | 4.76x |
| `list_mean_containment_to_complex` | 0.036853 s | 0.146917 s | 3.99x |

Все шесть сценариев прошли проверку равенства native/fallback. Времена описывают
один локальный прогон и не являются порогами для CI.

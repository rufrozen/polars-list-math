# `expected_value_of_game` performance

Сравнивает native и Python fallback для вычисления ожидаемого значения игры.
Complex-сценарий содержит 16 состояний и до трёх переходов из каждого состояния.

```bash
uv run pytest performance/native_plugins/expected_value_of_game -v -s
```

## Результат

Контрольный прогон 2026-08-22, 2 000 строк, 3 повтора:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `expected_value_of_game` | 0.004624 s | 0.012254 s | 2.65x |
| `expected_value_of_game_complex` | 0.023965 s | 0.077370 s | 3.23x |

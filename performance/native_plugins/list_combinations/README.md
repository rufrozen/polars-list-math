# List combinations performance

Сравнивает native и Python fallback для сгруппированных операций
`list_combinations` и `list_combinations_to`.
Complex-сценарии используют списки длиной 16 и 12 элементов с null,
индексами и фильтрацией null-пар.

```bash
uv run pytest performance/native_plugins/list_combinations -v -s
```

## Результаты

Контрольный прогон 2026-10-24, Python 3.12.3, Polars 2.0.0, WSL Linux x86_64,
release-сборка, 2 000 строк, 3 повтора. Типы и значения совпадают между реализациями.

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `list_combinations` | 0.000710 s | 0.021357 s | 30.07x |
| `list_combinations_to` | 0.000624 s | 0.015989 s | 25.62x |
| `list_combinations_complex` | 0.001466 s | 0.123840 s | 84.48x |
| `list_combinations_to_complex` | 0.001823 s | 0.156290 s | 85.74x |

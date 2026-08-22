# List combinations performance

Сравнивает native и Python fallback для сгруппированных операций
`list_combinations` и `list_combinations_to`.
Complex-сценарии используют списки длиной 16 и 12 элементов с null,
индексами и фильтрацией null-пар.

```bash
uv run pytest performance/native_plugins/list_combinations -v -s
```

## Результаты

Контрольный прогон 2026-08-22, 2 000 строк, 3 повтора:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `list_combinations` | 0.000498 s | 0.016221 s | 32.59x |
| `list_combinations_to` | 0.000860 s | 0.014249 s | 16.58x |
| `list_combinations_complex` | 0.006381 s | 0.122010 s | 19.12x |
| `list_combinations_to_complex` | 0.009351 s | 0.157177 s | 16.81x |

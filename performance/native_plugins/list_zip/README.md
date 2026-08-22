# `list_zip` performance

Сравнивает native и Python fallback для `list_zip`. Baseline использует две
короткие колонки; complex — четыре колонки разных типов со списками длиной до
24 элементов, разной длиной и padding.

```bash
uv run pytest performance/native_plugins/list_zip -v -s
```

## Результат

Контрольный прогон 2026-08-22, 2 000 строк, 3 повтора:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `list_zip` | 0.000394 s | 0.013344 s | 33.85x |
| `list_zip_complex` | 0.001124 s | 0.059398 s | 52.85x |

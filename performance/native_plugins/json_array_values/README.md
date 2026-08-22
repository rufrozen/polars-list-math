# `json_array_values` performance

Сравнивает native и Python fallback для разбора JSON-массивов. Complex-сценарий
содержит по 24 вложенных объекта с Unicode, списками и nullable metadata.

```bash
uv run pytest performance/native_plugins/json_array_values -v -s
```

## Результат

Контрольный прогон 2026-08-22, 2 000 строк, 3 повтора:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `json_array_values` | 0.001597 s | 0.012960 s | 8.12x |
| `json_array_values_complex` | 0.068937 s | 0.196068 s | 2.84x |

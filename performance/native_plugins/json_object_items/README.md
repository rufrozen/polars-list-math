# `json_object_items` performance

Сравнивает native и Python fallback для разбора JSON-объектов. Complex-сценарий
добавляет Unicode, вложенные объекты и массив из 20 объектов.

```bash
uv run pytest performance/native_plugins/json_object_items -v -s
```

## Результат

Контрольный прогон 2026-08-22, 2 000 строк, 3 повтора:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `json_object_items` | 0.003640 s | 0.018851 s | 5.18x |
| `json_object_items_complex` | 0.022595 s | 0.053219 s | 2.36x |

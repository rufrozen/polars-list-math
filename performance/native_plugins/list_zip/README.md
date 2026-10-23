# `list_zip` performance

Сравнивает native и Python fallback для `list_zip`. Baseline использует две
короткие колонки; complex — четыре колонки разных типов со списками длиной до
24 элементов, разной длиной и padding. Broadcast сочетает два списка разной
длины и скалярную валюту (включая null) с `broadcast=True, pad=True`.

```bash
uv run pytest performance/native_plugins/list_zip -v -s
```

## Результат

Контрольный прогон 2026-10-24, Python 3.12.3, Polars 2.0.0, WSL Linux x86_64,
release-сборка, 2 000 строк, 3 повтора. Типы и значения совпадают между реализациями.

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `list_zip` | 0.000499 s | 0.014638 s | 29.32x |
| `list_zip_complex` | 0.000837 s | 0.059503 s | 71.10x |
| `list_zip_broadcast` | 0.000806 s | 0.214209 s | 265.73x |

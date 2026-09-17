# `list_zip` performance

Сравнивает native и Python fallback для `list_zip`. Baseline использует две
короткие колонки; complex — четыре колонки разных типов со списками длиной до
24 элементов, разной длиной и padding. Broadcast сочетает два списка разной
длины и скалярную валюту (включая null) с `broadcast=True, pad=True`.

```bash
uv run pytest performance/native_plugins/list_zip -v -s
```

## Результат

Контрольный прогон 2026-09-18, 2 000 строк, 3 повтора, dev-сборка (`make develop`):

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `list_zip` | 0.002266 s | 0.013297 s | 5.87x |
| `list_zip_complex` | 0.007798 s | 0.058415 s | 7.49x |
| `list_zip_broadcast` | 0.009082 s | 0.056700 s | 6.24x |

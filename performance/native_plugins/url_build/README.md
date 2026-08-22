# `url_build` performance

Сравнивает native и Python fallback для сборки URL из шести компонентов.
Complex-тест использует длинные компоненты, Unicode, credentials, port,
параметры, повторяющиеся query-поля и fragment.

```bash
uv run pytest performance/native_plugins/url_build -v -s
```

## Результат

Контрольный прогон 2026-08-22, 2 000 строк, 3 повтора:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `url_build` | 0.001009 s | 0.005559 s | 5.51x |
| `url_build_complex` | 0.001187 s | 0.006419 s | 5.41x |

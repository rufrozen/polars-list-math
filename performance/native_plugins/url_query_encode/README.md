# `url_query_encode` performance

Сравнивает native и Python fallback для кодирования query string. Complex-тест
кодирует Unicode, специальные символы, URL и 12 значений одного параметра с
`doseq=True`.

```bash
uv run pytest performance/native_plugins/url_query_encode -v -s
```

## Результат

Контрольный прогон 2026-08-22, 2 000 строк, 3 повтора:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `url_query_encode` | 0.000912 s | 0.006108 s | 6.70x |
| `url_query_encode_complex` | 0.006447 s | 0.029197 s | 4.53x |

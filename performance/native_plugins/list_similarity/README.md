# List similarity performance

Сравнивает native и Python fallback для сгруппированных операций
`list_similarity`, `list_mean_similarity` и `list_mean_similarity_to`.
Complex-сценарии используют списки длиной до 32 элементов и вложенные наборы
из 8 списков, сравниваемые с 6 reference-списками.

```bash
uv run pytest performance/native_plugins/list_similarity -v -s
```

## Результаты

Контрольный прогон 2026-08-22, 2 000 строк, 3 повтора:

| Операция | Native | Python fallback | Ускорение |
|---|---:|---:|---:|
| `list_similarity` | 0.001043 s | 0.008305 s | 7.96x |
| `list_mean_similarity` | 0.002363 s | 0.043958 s | 18.60x |
| `list_mean_similarity_to` | 0.004872 s | 0.076319 s | 15.66x |
| `list_similarity_complex` | 0.010631 s | 0.048436 s | 4.56x |
| `list_mean_similarity_complex` | 0.038307 s | 0.580963 s | 15.17x |
| `list_mean_similarity_to_complex` | 0.060353 s | 0.940764 s | 15.59x |

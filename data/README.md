# Datos utilizados en los experimentos

Esta carpeta contiene **exactamente** los conjuntos con los que se probaron ABC y PSO.

## Base de datos

`experimentos.db` (SQLite) agrupa datasets, catálogo y resultados.

```bash
sqlite3 data/experimentos.db ".tables"
sqlite3 data/experimentos.db "SELECT * FROM catalogo_experimentos;"
sqlite3 data/experimentos.db "SELECT * FROM vista_resumen;"
```

Tablas:
- `dataset_breast_cancer` (569 filas, 30 variables + target)
- `dataset_wine` (178 filas, 13 variables + target)
- `dataset_iris` (150 filas, 4 variables + target)
- `dataset_xor` (4 filas)
- `dataset_blobs` (360 filas, 2 variables, K=4, semilla 42)
- `catalogo_experimentos` — qué se corrió, con qué archivo y semilla
- `resultados_metricas` — accuracy, C, gamma, silueta, etc.
- `features_seleccionadas` — variables que retuvo ABC
- `vista_resumen` — join catálogo + métricas

## CSV (mismos datos, abiertos)

| Archivo | Uso |
|---|---|
| `breast_cancer.csv` | ABC feature selection |
| `wine.csv` | ABC (corroboración) + PSO hiperparámetros + clustering 2D |
| `iris.csv` | MLP 4-8-4-3 entrenada con PSO |
| `xor.csv` | MLP 2-4-1 (frontera visual) |
| `blobs.csv` | PSO clustering vs K-Means |
| `catalogo_experimentos.json` | índice de los 7 ensayos |

Origen: conjuntos clásicos de UCI / scikit-learn, exportados para que el repositorio no dependa de descargar nada.

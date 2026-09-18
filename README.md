# Algoritmos de enjambre aplicados al ML

**Universidad Nacional del Altiplano de Puno (UNAP)**  
FIMEES — Escuela Profesional de Ingeniería de Sistemas  
Asignatura: Aprendizaje de Máquina (IX ciclo, grupo B)  
Autor: **Paolo Boris Luna Luque**  
Docente: Mayenka Fernandez Chambi

Repositorio: https://github.com/paololuna-luw/algoritmos-enjambre

## Qué hay aquí

| # | Tarea | Algoritmo | Dataset |
|---|---|---|---|
| 1 | Feature selection | ABC binario | Breast Cancer (30 var) + Wine |
| 2 | Hyperparameter tuning | PSO | SVM-RBF sobre Wine |
| 3 | Entrenar MLP sin backpropagation | PSO | Iris 4-8-4-3 + XOR 2-4-1 |
| 4 | Clustering | PSO | Blobs K=4 + Wine 2D |

Informe: [`report/informe.pdf`](report/informe.pdf)

## Cómo reproducir

```bash
python -m pip install -r requirements.txt
python src/experiments.py
```

Figuras en `figures/`. Resumen numérico en `figures/results.json`.

## Ciclo del enjambre (los cuatro ejemplos)

1. Representación de la partícula / fuente de alimento
2. Inicialización
3. Función de aptitud
4. Comportamiento
5. Evolución
6. Criterio de parada

## Resultados (semilla 42)

- ABC / Breast Cancer: 30 → 13 variables; accuracy CV 0.975 → **0.981** (SelectKBest k=13: 0.949)
- ABC / Wine: 13 → 6 variables; 0.978 → **0.983**
- PSO-SVM / Wine: C≈92.3, γ≈0.024; test **0.981** (grid 0.963)
- PSO-MLP Iris 4-8-4-3 (91 pesos, sin gradiente): test **0.911** vs Adam 0.756
- XOR MSE ≈ 8e-5
- PSO-clustering: silueta 0.610 (blobs) vs K-Means 0.824; Wine-2D empate 0.484


## Datos (lo que se probó)

Los datasets y los resultados quedan en [`data/`](data/):

- `data/experimentos.db` — base SQLite con datasets + catálogo + métricas
- `data/*.csv` — los mismos conjuntos en CSV
- `data/catalogo_experimentos.json` — índice de cada ensayo

```bash
sqlite3 data/experimentos.db "SELECT id, tarea, dataset, n_filas FROM catalogo_experimentos;"
sqlite3 data/experimentos.db "SELECT * FROM vista_resumen;"
```

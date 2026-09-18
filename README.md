# Algoritmos de enjambre aplicados al aprendizaje automático

**Universidad Nacional del Altiplano (UNAP)**  
Facultad de Ingeniería Mecánica Eléctrica, Electrónica y de Sistemas (FIMEES)  
Escuela Profesional de Ingeniería de Sistemas

| | |
|---|---|
| Curso | Aprendizaje de Máquina |
| Ciclo / grupo | IX ciclo, grupo B |
| Autor | Paolo Boris Luna Luque (código 227631) |
| Docente | Mayenka Fernandez Chambi |
| Repositorio | https://github.com/paololuna-luw/algoritmos-enjambre |
| Video | https://youtu.be/quXcSntND30 |

Actividad 03. Cuatro problemas de aprendizaje automático resueltos con colonia artificial de abejas (ABC) y enjambre de partículas (PSO). En cada ejemplo se recorre el mismo ciclo: representación, inicialización, aptitud, conducta, evolución y parada.

---

## Contenido del repositorio

```
algoritmos-enjambre/
├── codes/                 scripts Python (motor + cuatro ejercicios)
├── notebooks/             cuadernos listos para Google Colab
├── data/                  CSV, SQLite y catálogo de lo que se probó
├── figures/               gráficos del informe y del video
├── presentation_html/     pantallas de la exposición
├── requirements.txt
└── README.md
```

| Carpeta | Qué contiene |
|---|---|
| `codes/` | `enjambre.py` (ABC y PSO), `01`–`04_*.py`, `experiments.py` |
| `notebooks/` | un `.ipynb` por ejercicio; la primera celda trae el motor |
| `data/` | CSV de Breast Cancer, Wine, Iris, XOR y blobs, más `experimentos.db` |
| `figures/` | máscaras, heatmaps, fronteras, convergencias y `results.json` |
| `presentation_html/` | `index.html` y las cuatro misiones del video |

El informe PDF se entrega en el aula virtual. No forma parte de este repositorio.

---

## Los cuatro ejercicios

| # | Problema | Buscador | Modelo | Datos | Control |
|---|---|---|---|---|---|
| 1 | Selección de características | ABC binario | Regresión logística | Breast Cancer (30 var.) y Wine (13) | SelectKBest, mismo *k* |
| 2 | Ajuste de hiperparámetros | PSO | SVM-RBF | Wine | Grid 5×5 y SVM por defecto |
| 3 | Pesos de una red sin retropropagación | PSO | MLP 4-8-4-3 (91 params.) | Iris; XOR 2-4-1 de apoyo | Adam, 400 épocas |
| 4 | Agrupamiento | PSO | Centroides / inercia | Blobs *K*=4; Wine 2D *K*=3 | K-Means `n_init=10` |

Tres piezas que no se mezclan: el **dato** (tabla), el **modelo** (clasificador o regla de agrupamiento) y el **buscador** (ABC o PSO). En el ejercicio 2 no se eligen columnas del vino; se buscan *C* y *γ* del SVM.

---

## Cómo reproducir

```bash
git clone https://github.com/paololuna-luw/algoritmos-enjambre.git
cd algoritmos-enjambre
python -m pip install -r requirements.txt
```

Scripts locales (leen `data/*.csv` y escriben en `figures/`):

```bash
python codes/01_abc_feature_selection.py
python codes/02_pso_hyperparameter_tuning.py
python codes/03_pso_entrenar_red.py
python codes/04_pso_clustering.py
```

O, de una vez: `python codes/experiments.py`.

**Colab.** Abrir cualquier archivo de `notebooks/` → *Runtime → Run all*. No hace falta subir CSV: cargan `sklearn.datasets`. Semilla **42**.

**Video.** Abrir `presentation_html/index.html` en el navegador.

**Base de lo que se probó:**

```bash
sqlite3 data/experimentos.db "SELECT id, tarea, dataset, n_filas FROM catalogo_experimentos;"
sqlite3 data/experimentos.db "SELECT * FROM vista_resumen;"
```

---

## Resultados (semilla 42)

| Ejercicio | Hallazgo |
|---|---|
| ABC / Breast Cancer | 30 → 13 variables. Accuracy CV 0.975 → **0.981**. SelectKBest con *k*=13: **0.949** |
| ABC / Wine | 13 → 6 variables. 0.978 → **0.983** |
| PSO-SVM / Wine | *C* ≈ 92.3, *γ* ≈ 0.024. Test **0.981** (grid 0.963; default 0.981) |
| PSO-MLP / Iris | Test **0.911** frente a Adam 0.756 (sin convergencia a 400 épocas) |
| XOR | MSE ≈ 8×10⁻⁵ |
| Clustering / blobs | Silueta PSO **0.610**, K-Means **0.824** |
| Clustering / Wine 2D | Ambos **0.484** |

SelectKBest no es una muestra aleatoria: ordena por relevancia individual (ANOVA). A igual cardinalidad, el wrapper de ABC obtiene mejor equipo de variables.

K-Means gana en las nubes sintéticas. El ejercicio 4 documenta el ciclo, no un recuento favorable.

---

## Ciclo (los cuatro ejemplos)

1. Representación de la partícula o de la fuente de alimento
2. Inicialización del enjambre
3. Función de aptitud
4. Conducta del agente
5. Evolución
6. Criterio de parada

ABC opera en {0,1}^d (máscara de columnas). PSO opera en R^d: par (log C, log γ), vector de 91 pesos, o K centroides concatenados.

---

Material elaborado para la Actividad 03 del curso Aprendizaje de Máquina, UNAP, septiembre de 2026.

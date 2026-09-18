#!/usr/bin/env python3
"""Ejercicio 2 — Hyperparameter tuning con Particle Swarm Optimization (PSO).

Dataset: data/wine.csv
Particula: (log10 C, log10 gamma) de un SVM RBF.
Aptitud: 1 - accuracy CV (se minimiza).
Baseline: SVM default y Grid Search 5x5.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from enjambre import pso_minimize

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 160, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    print("=" * 60)
    print("EJERCICIO 2  PSO — Hyperparameter tuning (SVM-RBF)")
    print("=" * 60)

    df = pd.read_csv(DATA / "wine.csv")
    X = StandardScaler().fit_transform(df.drop(columns=["target", "target_name"]).values)
    y = df["target"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(p):
        C, g = 10 ** float(p[0]), 10 ** float(p[1])
        acc = cross_val_score(SVC(C=C, gamma=g, kernel="rbf"), Xtr, ytr, cv=inner, scoring="accuracy").mean()
        return float(1.0 - acc)

    bounds = (np.array([-2.0, -4.0]), np.array([3.0, 1.0]))
    best, cost, hist, swarm = pso_minimize(objective, bounds, n_particles=10, iters=10, seed=SEED)
    C_star, g_star = 10 ** best[0], 10 ** best[1]
    acc_pso_te = accuracy_score(yte, SVC(C=C_star, gamma=g_star, kernel="rbf").fit(Xtr, ytr).predict(Xte))
    acc_pso_cv = 1.0 - cost

    grid = GridSearchCV(
        SVC(kernel="rbf"),
        {"C": np.logspace(-2, 3, 5), "gamma": np.logspace(-4, 1, 5)},
        cv=inner, scoring="accuracy",
    ).fit(Xtr, ytr)
    acc_grid_te = accuracy_score(yte, grid.predict(Xte))
    acc_default = accuracy_score(yte, SVC().fit(Xtr, ytr).predict(Xte))

    print(f"PSO    C={C_star:.3f}  gamma={g_star:.5f}  CV={acc_pso_cv:.4f}  test={acc_pso_te:.4f}")
    print(f"Grid   C={grid.best_params_['C']:.3f}  gamma={grid.best_params_['gamma']:.5f}  test={acc_grid_te:.4f}")
    print(f"Default test={acc_default:.4f}")

    Cs = np.linspace(-2, 3, 7); Gs = np.linspace(-4, 1, 7)
    Z = np.zeros((len(Gs), len(Cs)))
    for i, gv in enumerate(Gs):
        for j, cv_ in enumerate(Cs):
            Z[i, j] = objective([cv_, gv])

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.contourf(Cs, Gs, Z, levels=12, cmap="viridis_r")
    fig.colorbar(im, ax=ax, label="1 - accuracy CV")
    last = swarm[-1]
    ax.scatter(last[:, 0], last[:, 1], c="#f4d35e", s=28, edgecolor="k", linewidth=0.4, label="enjambre final")
    ax.scatter(best[0], best[1], c="red", s=90, marker="*", zorder=5, label="gbest PSO")
    ax.scatter(np.log10(grid.best_params_["C"]), np.log10(grid.best_params_["gamma"]),
               c="white", s=50, marker="D", edgecolor="k", label="Grid Search")
    ax.set_xlabel(r"$\log_{10} C$"); ax.set_ylabel(r"$\log_{10} \gamma$")
    ax.set_title("PSO en el espacio de hiperparametros SVM-RBF (Wine)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "02_pso_heatmap.png", bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.plot(hist, color="#1f6aa5", marker="o", ms=3)
    ax.set_xlabel("Iteracion"); ax.set_ylabel("Mejor error CV (1 - acc)")
    ax.set_title("Convergencia PSO — ajuste de hiperparametros")
    fig.tight_layout(); fig.savefig(FIG / "02_pso_convergence.png", bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    labs = ["SVM default", "Grid Search\n(25 evals)", "PSO\n(10x10 evals)"]
    vals = [acc_default, acc_grid_te, acc_pso_te]
    bars = ax.bar(labs, vals, color=["#888888", "#4c78a8", "#1f6aa5"])
    ax.set_ylim(0.7, 1.05); ax.set_ylabel("Accuracy en test (30%)")
    ax.set_title("Wine — SVM-RBF: default vs Grid vs PSO")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(FIG / "02_pso_compare.png", bbox_inches="tight"); plt.close()

    out = {
        "dataset": "Wine",
        "C": float(C_star), "gamma": float(g_star),
        "acc_cv_pso": float(acc_pso_cv), "acc_test_pso": float(acc_pso_te),
        "acc_test_grid": float(acc_grid_te), "acc_test_default": float(acc_default),
        "grid_best": {k: float(v) for k, v in grid.best_params_.items()},
        "pso_evals": 10 * 11, "grid_evals": 25,
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()

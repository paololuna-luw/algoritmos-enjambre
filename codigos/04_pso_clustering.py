#!/usr/bin/env python3
"""Ejercicio 4 — Clustering con Particle Swarm Optimization.

data/blobs.csv  : 360 puntos, K=4 (sintético, semilla 42)
data/wine.csv   : primeras 2 variables estandarizadas, K=3
Particula: K centroides concatenados.
Aptitud: inercia (suma de distancias al cuadrado al centro mas cercano).
Baseline: K-Means n_init=10.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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


def assign(X, C):
    d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return d.argmin(1), float(d.min(axis=1).sum())


def main():
    print("=" * 60)
    print("EJERCICIO 4  PSO — Clustering")
    print("=" * 60)

    blobs = pd.read_csv(DATA / "blobs.csv")
    X = blobs[["x1", "x2"]].values
    K = 4

    def inertia(pos):
        C = pos.reshape(K, 2)
        _, J = assign(X, C)
        return J

    bounds = (np.repeat(X.min(axis=0), K), np.repeat(X.max(axis=0), K))
    best, cost, hist, _ = pso_minimize(inertia, bounds, n_particles=18, iters=22, seed=SEED)
    C_pso = best.reshape(K, 2)
    lab_pso, _ = assign(X, C_pso)
    sil_pso = float(silhouette_score(X, lab_pso))

    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
    sil_km = float(silhouette_score(X, km.labels_))
    print(f"Blobs  inercia PSO={cost:.1f}  KM={km.inertia_:.1f}")
    print(f"       silueta PSO={sil_pso:.3f}  KM={sil_km:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True, sharey=True)
    ax[0].scatter(X[:, 0], X[:, 1], c=lab_pso, cmap="tab10", s=14)
    ax[0].scatter(C_pso[:, 0], C_pso[:, 1], c="k", marker="X", s=90)
    ax[0].set_title(f"PSO-clustering  K=4   sil={sil_pso:.3f}")
    ax[1].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="tab10", s=14)
    ax[1].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c="k", marker="X", s=90)
    ax[1].set_title(f"K-Means  K=4   sil={sil_km:.3f}")
    fig.tight_layout(); fig.savefig(FIG / "04_clusters.png", bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(hist, color="#6a4c93")
    ax.set_xlabel("Iteracion"); ax.set_ylabel("Inercia")
    ax.set_title("PSO — evolucion de la inercia de clustering")
    fig.tight_layout(); fig.savefig(FIG / "04_cluster_convergence.png", bbox_inches="tight"); plt.close()

    wine = pd.read_csv(DATA / "wine.csv")
    Xw = StandardScaler().fit_transform(wine.iloc[:, :2].values)
    names = list(wine.columns[:2])
    Kw = 3

    def inertia_w(pos):
        C = pos.reshape(Kw, 2)
        _, J = assign(Xw, C)
        return J

    bw = (np.repeat(Xw.min(0), Kw), np.repeat(Xw.max(0), Kw))
    bestw, costw, _, _ = pso_minimize(inertia_w, bw, n_particles=14, iters=18, seed=3)
    Cw = bestw.reshape(Kw, 2)
    labw, _ = assign(Xw, Cw)
    silw = float(silhouette_score(Xw, labw))
    kmw = KMeans(n_clusters=Kw, n_init=10, random_state=SEED).fit(Xw)
    sil_kmw = float(silhouette_score(Xw, kmw.labels_))
    print(f"Wine2D silueta PSO={silw:.3f}  KM={sil_kmw:.3f}")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(Xw[:, 0], Xw[:, 1], c=labw, cmap="tab10", s=18)
    ax.scatter(Cw[:, 0], Cw[:, 1], c="k", marker="X", s=90, label="centroides PSO")
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
    ax.set_title(f"Wine (2 vars) — PSO clustering  sil={silw:.3f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "04_wine_clusters.png", bbox_inches="tight"); plt.close()

    out = {
        "blobs_inertia_pso": float(cost),
        "blobs_inertia_kmeans": float(km.inertia_),
        "blobs_sil_pso": sil_pso,
        "blobs_sil_kmeans": sil_km,
        "wine2_sil_pso": silw,
        "wine2_sil_kmeans": sil_kmw,
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()

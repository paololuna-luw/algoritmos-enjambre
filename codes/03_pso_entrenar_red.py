#!/usr/bin/env python3
"""Ejercicio 3 — Entrenar una red neuronal SIN backpropagation, con PSO.

Iris: MLP 4-8-4-3 (91 pesos + bias). Cada particula ES una red completa.
XOR:  MLP 2-4-1 para ver la frontera de decision.
No se calcula ningun gradiente: solo forward + aptitud.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def relu(z):
    return np.maximum(0.0, z)


def pack_shapes(dims):
    specs, n = [], 0
    for a, b in zip(dims[:-1], dims[1:]):
        specs.append(((a, b), (b,)))
        n += a * b + b
    return specs, n


def unpack(vec, specs):
    params, i = [], 0
    for ws, bs in specs:
        nw = int(np.prod(ws))
        W = vec[i:i + nw].reshape(ws); i += nw
        B = vec[i:i + bs[0]]; i += bs[0]
        params.append((W, B))
    return params


def forward_mlp(X, params, last="softmax"):
    h = X
    for i, (W, B) in enumerate(params):
        h = h @ W + B
        if i < len(params) - 1:
            h = relu(h)
        else:
            h = softmax(h) if last == "softmax" else 1 / (1 + np.exp(-h))
    return h


def main():
    print("=" * 60)
    print("EJERCICIO 3  PSO — MLP sin backpropagation")
    print("=" * 60)

    df = pd.read_csv(DATA / "iris.csv")
    X = StandardScaler().fit_transform(df.drop(columns=["target", "target_name"]).values)
    y = df["target"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    dims = [4, 8, 4, 3]
    specs, n_dim = pack_shapes(dims)
    print(f"Arquitectura {dims}  parametros={n_dim}")

    def objective(w):
        pred = forward_mlp(Xtr, unpack(w, specs))
        return float(-np.mean(np.log(pred[np.arange(len(ytr)), ytr] + 1e-8)))

    bounds = (np.full(n_dim, -2.5), np.full(n_dim, 2.5))
    best, cost, hist, _ = pso_minimize(
        objective, bounds, n_particles=18, iters=25, w=0.65, c1=1.6, c2=1.6, seed=SEED
    )
    params = unpack(best, specs)
    acc_tr = float((forward_mlp(Xtr, params).argmax(1) == ytr).mean())
    acc_te = float((forward_mlp(Xte, params).argmax(1) == yte).mean())

    mlp = MLPClassifier(hidden_layer_sizes=(8, 4), activation="relu", max_iter=400,
                        random_state=SEED, solver="adam")
    mlp.fit(Xtr, ytr)
    acc_bp_te = float(mlp.score(Xte, yte))
    print(f"PSO     train={acc_tr:.4f}  test={acc_te:.4f}  CE={cost:.4f}")
    print(f"Adam    test={acc_bp_te:.4f}")

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.plot(hist, color="#2a9d8f")
    ax.set_xlabel("Iteracion PSO"); ax.set_ylabel("Cross-entropy (train)")
    ax.set_title("Entrenamiento MLP 4-8-4-3 con PSO (sin backpropagation)")
    fig.tight_layout(); fig.savefig(FIG / "03_nn_convergence.png", bbox_inches="tight"); plt.close()

    # XOR visual
    xor = pd.read_csv(DATA / "xor.csv")
    Xor, yor = xor[["x1", "x2"]].values, xor["y"].values
    dims_x = [2, 4, 1]
    specs_x, ndx = pack_shapes(dims_x)

    def obj_xor(w):
        p = forward_mlp(Xor, unpack(w, specs_x), last="sigmoid").ravel()
        return float(np.mean((p - yor) ** 2))

    bestx, costx, histx, _ = pso_minimize(
        obj_xor, (np.full(ndx, -4.0), np.full(ndx, 4.0)), n_particles=16, iters=30, seed=1
    )
    prx = unpack(bestx, specs_x)
    xor_preds = forward_mlp(Xor, prx, last="sigmoid").ravel().tolist()
    print(f"XOR MSE={costx:.6f}  preds={np.round(xor_preds, 3).tolist()}")

    xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 160), np.linspace(-0.3, 1.3, 160))
    zz = forward_mlp(np.c_[xx.ravel(), yy.ravel()], prx, last="sigmoid").reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", vmin=0, vmax=1)
    ax.scatter(Xor[:, 0], Xor[:, 1], c=yor, cmap="RdBu", s=80, edgecolor="k")
    ax.set_title("XOR — frontera de la MLP 2-4-1 entrenada con PSO")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    fig.tight_layout(); fig.savefig(FIG / "03_xor_boundary.png", bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    vals = [acc_te, acc_bp_te]
    bars = ax.bar(["PSO-MLP test", "Adam/backprop test"], vals, color=["#2a9d8f", "#4c78a8"])
    ax.set_ylim(0, 1.15); ax.set_ylabel("Accuracy")
    ax.set_title("Iris — MLP 4-8-4-3: PSO vs backpropagation")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(FIG / "03_nn_compare.png", bbox_inches="tight"); plt.close()

    out = {
        "architecture": "4-8-4-3", "n_parameters": int(n_dim),
        "ce_train": float(cost), "acc_train_pso": acc_tr, "acc_test_pso": acc_te,
        "acc_test_backprop": acc_bp_te, "xor_mse": float(costx), "xor_preds": xor_preds,
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()

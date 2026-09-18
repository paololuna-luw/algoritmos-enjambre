#!/usr/bin/env python3
"""Ejercicio 1 — Feature selection con Artificial Bee Colony (ABC).

Dataset: data/breast_cancer.csv (30 variables) y data/wine.csv (13 variables).
Particula / fuente de alimento: mascara binaria (1 = usar la variable).
Aptitud: accuracy CV de LogisticRegression menos penalizacion por cardinalidad.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from enjambre import abc_binary_maximize

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


def load_xy(csv_name, target="target"):
    df = pd.read_csv(DATA / csv_name)
    drop = [c for c in (target, "target_name") if c in df.columns]
    X = StandardScaler().fit_transform(df.drop(columns=drop).values)
    y = df[target].values
    names = [c for c in df.columns if c not in drop]
    return X, y, np.array(names)


def fitness_factory(X, y, clf, cv, lam):
    def eval_mask(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        acc = cross_val_score(clf, X[:, mask], y, cv=cv, scoring="accuracy").mean()
        return float(acc - lam * (mask.sum() / mask.size))
    return eval_mask


def main():
    print("=" * 60)
    print("EJERCICIO 1  ABC — Feature selection")
    print("=" * 60)

    # --- Breast Cancer ---
    X, y, names = load_xy("breast_cancer.csv")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    fit_fn = fitness_factory(X, y, clf, cv, lam=0.012)

    acc_all = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    mask, best_fit, hist = abc_binary_maximize(
        fit_fn, n_bits=X.shape[1], n_bees=8, cycles=8, limit=4, seed=SEED
    )
    acc_abc = cross_val_score(clf, X[:, mask.astype(bool)], y, cv=cv, scoring="accuracy").mean()
    k = max(int(mask.sum()), 1)
    acc_skb = cross_val_score(
        clf, SelectKBest(f_classif, k=k).fit_transform(X, y), y, cv=cv, scoring="accuracy"
    ).mean()
    selected = names[mask.astype(bool)].tolist()

    print(f"Breast Cancer  30 -> {int(mask.sum())} variables")
    print(f"  Acc todas={acc_all:.4f}  ABC={acc_abc:.4f}  SelectKBest={acc_skb:.4f}")
    print("  Variables ABC:", ", ".join(selected))

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ax[0].plot(hist, marker="o", ms=3, color="#c47b16")
    ax[0].set_xlabel("Ciclo"); ax[0].set_ylabel("Fitness")
    ax[0].set_title("ABC — convergencia (Breast Cancer)")
    ax[1].barh(np.arange(len(names)), mask, color=np.where(mask, "#c47b16", "#dddddd"))
    ax[1].set_yticks(np.arange(len(names))); ax[1].set_yticklabels(names, fontsize=6)
    ax[1].set_title("Mascara de caracteristicas ABC")
    fig.tight_layout(); fig.savefig(FIG / "01_abc_features.png", bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = ["Todas\n(30)", f"ABC\n({int(mask.sum())})", f"SelectKBest\n(k={k})"]
    vals = [acc_all, acc_abc, acc_skb]
    bars = ax.bar(labels, vals, color=["#888888", "#c47b16", "#4c78a8"])
    ax.set_ylim(min(vals) - 0.03, 1.0); ax.set_ylabel("Accuracy CV (3-fold)")
    ax.set_title("Breast Cancer — seleccion de caracteristicas")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center")
    fig.tight_layout(); fig.savefig(FIG / "01_abc_compare.png", bbox_inches="tight"); plt.close()

    # --- Wine (corroboracion) ---
    Xw, yw, names_w = load_xy("wine.csv")
    clfw = LogisticRegression(max_iter=250)
    fit_w = fitness_factory(Xw, yw, clfw, 3, lam=0.015)
    acc_w_all = cross_val_score(clfw, Xw, yw, cv=3, scoring="accuracy").mean()
    mw, fw, hw = abc_binary_maximize(fit_w, n_bits=Xw.shape[1], n_bees=8, cycles=8, seed=7)
    acc_w_abc = cross_val_score(clfw, Xw[:, mw.astype(bool)], yw, cv=3, scoring="accuracy").mean()
    wine_sel = names_w[mw.astype(bool)].tolist()
    print(f"Wine           13 -> {int(mw.sum())} variables")
    print(f"  Acc todas={acc_w_all:.4f}  ABC={acc_w_abc:.4f}")
    print("  Variables ABC:", ", ".join(wine_sel))

    out = {
        "dataset": "Breast Cancer Wisconsin",
        "n_features_total": int(X.shape[1]),
        "n_features_abc": int(mask.sum()),
        "features_abc": selected,
        "acc_all": float(acc_all),
        "acc_abc": float(acc_abc),
        "acc_selectkbest": float(acc_skb),
        "best_fitness": float(best_fit),
        "wine_n_total": int(Xw.shape[1]),
        "wine_n_abc": int(mw.sum()),
        "wine_features": wine_sel,
        "wine_acc_all": float(acc_w_all),
        "wine_acc_abc": float(acc_w_abc),
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()

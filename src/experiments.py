#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_blobs
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 160, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def pso_minimize(objective, bounds, n_particles=20, iters=25, w=0.7, c1=1.5, c2=1.5, seed=42):
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    dim = lo.size
    pos = rng.uniform(lo, hi, size=(n_particles, dim))
    vel = np.zeros_like(pos)
    costs = np.array([objective(p) for p in pos])
    pbest, pbest_c = pos.copy(), costs.copy()
    g = int(np.argmin(pbest_c))
    gbest, gbest_c = pbest[g].copy(), float(pbest_c[g])
    hist = [gbest_c]
    swarm_hist = [pos.copy()]
    for _ in range(iters):
        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        costs = np.array([objective(p) for p in pos])
        improved = costs < pbest_c
        pbest[improved] = pos[improved]
        pbest_c[improved] = costs[improved]
        g = int(np.argmin(pbest_c))
        if pbest_c[g] < gbest_c:
            gbest, gbest_c = pbest[g].copy(), float(pbest_c[g])
        hist.append(gbest_c)
        swarm_hist.append(pos.copy())
    return gbest, gbest_c, np.array(hist), swarm_hist


def abc_binary_maximize(objective, n_bits, n_bees=10, cycles=12, limit=4, seed=42):
    rng = np.random.default_rng(seed)
    foods = rng.integers(0, 2, size=(n_bees, n_bits))
    empty = foods.sum(axis=1) == 0
    if empty.any():
        foods[empty, rng.integers(0, n_bits, size=int(empty.sum()))] = 1
    fit = np.array([objective(f) for f in foods])
    trials = np.zeros(n_bees, dtype=int)
    best_i = int(np.argmax(fit))
    best, best_f = foods[best_i].copy(), float(fit[best_i])
    hist = [best_f]

    def neighbor(src):
        k = int(rng.integers(0, n_bits))
        nxt = src.copy()
        nxt[k] ^= 1
        if nxt.sum() == 0:
            nxt[k] = 1
        return nxt

    for _ in range(cycles):
        for i in range(n_bees):
            cand = neighbor(foods[i])
            fc = objective(cand)
            if fc >= fit[i]:
                foods[i], fit[i], trials[i] = cand, fc, 0
            else:
                trials[i] += 1
        probs = fit - fit.min() + 1e-9
        probs = probs / probs.sum()
        for _o in range(n_bees):
            i = int(rng.choice(n_bees, p=probs))
            cand = neighbor(foods[i])
            fc = objective(cand)
            if fc >= fit[i]:
                foods[i], fit[i], trials[i] = cand, fc, 0
            else:
                trials[i] += 1
        for i in range(n_bees):
            if trials[i] >= limit:
                foods[i] = rng.integers(0, 2, size=n_bits)
                if foods[i].sum() == 0:
                    foods[i][int(rng.integers(0, n_bits))] = 1
                fit[i] = objective(foods[i])
                trials[i] = 0
        bi = int(np.argmax(fit))
        if fit[bi] > best_f:
            best, best_f = foods[bi].copy(), float(fit[bi])
        hist.append(best_f)
    return best, best_f, np.array(hist)


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
        W = vec[i:i + nw].reshape(ws)
        i += nw
        B = vec[i:i + bs[0]]
        i += bs[0]
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


def run_feature_selection():
    data = load_breast_cancer()
    X, y = StandardScaler().fit_transform(data.data), data.target
    names = np.array(data.feature_names)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=200, solver="liblinear")

    def eval_mask(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        acc = cross_val_score(clf, X[:, mask], y, cv=cv, scoring="accuracy").mean()
        return float(acc - 0.012 * (mask.sum() / mask.size))

    acc_all = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    mask, fit, hist = abc_binary_maximize(eval_mask, n_bits=X.shape[1], n_bees=8, cycles=8, limit=4, seed=SEED)
    selected = names[mask.astype(bool)]
    acc_sel = cross_val_score(clf, X[:, mask.astype(bool)], y, cv=cv, scoring="accuracy").mean()
    k = max(int(mask.sum()), 1)
    Xs = SelectKBest(f_classif, k=k).fit_transform(X, y)
    acc_skb = cross_val_score(clf, Xs, y, cv=cv, scoring="accuracy").mean()

    wine = load_wine()
    Xw, yw = StandardScaler().fit_transform(wine.data), wine.target
    clfw = LogisticRegression(max_iter=250)

    def eval_wine(mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        acc = cross_val_score(clfw, Xw[:, mask], yw, cv=3, scoring="accuracy").mean()
        return float(acc - 0.015 * (mask.sum() / mask.size))

    acc_wine_all = cross_val_score(clfw, Xw, yw, cv=3, scoring="accuracy").mean()
    mw, fw, hw = abc_binary_maximize(eval_wine, n_bits=Xw.shape[1], n_bees=8, cycles=8, seed=7)
    acc_wine_sel = cross_val_score(clfw, Xw[:, mw.astype(bool)], yw, cv=3, scoring="accuracy").mean()

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
    vals = [acc_all, acc_sel, acc_skb]
    bars = ax.bar(labels, vals, color=["#888888", "#c47b16", "#4c78a8"])
    ax.set_ylim(min(vals) - 0.03, 1.0); ax.set_ylabel("Accuracy CV (3-fold)")
    ax.set_title("Breast Cancer — seleccion de caracteristicas")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "01_abc_compare.png", bbox_inches="tight"); plt.close()

    return {
        "dataset": "Breast Cancer Wisconsin",
        "n_features_total": int(X.shape[1]),
        "n_features_abc": int(mask.sum()),
        "features_abc": selected.tolist(),
        "acc_all": float(acc_all),
        "acc_abc": float(acc_sel),
        "acc_selectkbest": float(acc_skb),
        "best_fitness": float(fit),
        "wine_n_total": int(Xw.shape[1]),
        "wine_n_abc": int(mw.sum()),
        "wine_features": [n for n, m in zip(wine.feature_names, mw.astype(bool)) if m],
        "wine_acc_all": float(acc_wine_all),
        "wine_acc_abc": float(acc_wine_sel),
    }


def run_hyperparams():
    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    y = data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(p):
        C = 10 ** float(p[0]); g = 10 ** float(p[1])
        acc = cross_val_score(SVC(C=C, gamma=g, kernel="rbf"), Xtr, ytr, cv=inner, scoring="accuracy").mean()
        return float(1.0 - acc)

    bounds = (np.array([-2.0, -4.0]), np.array([3.0, 1.0]))
    best, cost, hist, swarm = pso_minimize(objective, bounds, n_particles=10, iters=10, seed=SEED)
    C_star, g_star = 10 ** best[0], 10 ** best[1]
    clf_pso = SVC(C=C_star, gamma=g_star, kernel="rbf").fit(Xtr, ytr)
    acc_pso_te = accuracy_score(yte, clf_pso.predict(Xte))
    acc_pso_cv = 1.0 - cost

    grid = GridSearchCV(SVC(kernel="rbf"),
                        {"C": np.logspace(-2, 3, 5), "gamma": np.logspace(-4, 1, 5)},
                        cv=inner, scoring="accuracy")
    grid.fit(Xtr, ytr)
    acc_grid_te = accuracy_score(yte, grid.predict(Xte))
    acc_default = accuracy_score(yte, SVC().fit(Xtr, ytr).predict(Xte))

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

    return {
        "dataset": "Wine",
        "C": float(C_star), "gamma": float(g_star),
        "acc_cv_pso": float(acc_pso_cv), "acc_test_pso": float(acc_pso_te),
        "acc_test_grid": float(acc_grid_te), "acc_test_default": float(acc_default),
        "grid_best": {k: float(v) for k, v in grid.best_params_.items()},
        "pso_evals": 10 * 11, "grid_evals": 25,
    }


def run_nn():
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    y = iris.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    dims = [4, 8, 4, 3]
    specs, n_dim = pack_shapes(dims)

    def objective(w):
        pred = forward_mlp(Xtr, unpack(w, specs))
        return float(-np.mean(np.log(pred[np.arange(len(ytr)), ytr] + 1e-8)))

    bounds = (np.full(n_dim, -2.5), np.full(n_dim, 2.5))
    best, cost, hist, _ = pso_minimize(objective, bounds, n_particles=18, iters=25, w=0.65, c1=1.6, c2=1.6, seed=SEED)
    params = unpack(best, specs)
    acc_tr = float((forward_mlp(Xtr, params).argmax(1) == ytr).mean())
    acc_te = float((forward_mlp(Xte, params).argmax(1) == yte).mean())
    mlp = MLPClassifier(hidden_layer_sizes=(8, 4), activation="relu", max_iter=400,
                        random_state=SEED, solver="adam")
    mlp.fit(Xtr, ytr)
    acc_bp_te = float(mlp.score(Xte, yte))

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.plot(hist, color="#2a9d8f")
    ax.set_xlabel("Iteracion PSO"); ax.set_ylabel("Cross-entropy (train)")
    ax.set_title("Entrenamiento MLP 4-8-4-3 con PSO (sin backpropagation)")
    fig.tight_layout(); fig.savefig(FIG / "03_nn_convergence.png", bbox_inches="tight"); plt.close()

    Xor = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    yor = np.array([0.0, 1.0, 1.0, 0.0])
    dims_x = [2, 4, 1]
    specs_x, ndx = pack_shapes(dims_x)

    def obj_xor(w):
        p = forward_mlp(Xor, unpack(w, specs_x), last="sigmoid").ravel()
        return float(np.mean((p - yor) ** 2))

    bestx, costx, histx, _ = pso_minimize(obj_xor, (np.full(ndx, -4.0), np.full(ndx, 4.0)),
                                         n_particles=16, iters=30, seed=1)
    prx = unpack(bestx, specs_x)
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

    return {
        "architecture": "4-8-4-3", "n_parameters": int(n_dim),
        "ce_train": float(cost), "acc_train_pso": acc_tr, "acc_test_pso": acc_te,
        "acc_test_backprop": acc_bp_te, "xor_mse": float(costx),
        "xor_preds": forward_mlp(Xor, prx, last="sigmoid").ravel().tolist(),
    }


def run_clustering():
    X, _ = make_blobs(n_samples=360, centers=4, cluster_std=0.85, random_state=SEED)
    K = 4

    def inertia(pos):
        C = pos.reshape(K, 2)
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        return float(d.min(axis=1).sum())

    bounds = (np.repeat(X.min(axis=0), K), np.repeat(X.max(axis=0), K))
    best, cost, hist, _ = pso_minimize(inertia, bounds, n_particles=18, iters=22, seed=SEED)
    C_pso = best.reshape(K, 2)
    lab_pso = ((X[:, None, :] - C_pso[None, :, :]) ** 2).sum(axis=2).argmin(1)
    sil_pso = float(silhouette_score(X, lab_pso))
    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(X)
    sil_km = float(silhouette_score(X, km.labels_))

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

    wine = load_wine()
    Xw = StandardScaler().fit_transform(wine.data[:, :2])
    Kw = 3

    def inertia_w(pos):
        C = pos.reshape(Kw, 2)
        d = ((Xw[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        return float(d.min(axis=1).sum())

    bw = (np.repeat(Xw.min(0), Kw), np.repeat(Xw.max(0), Kw))
    bestw, costw, _, _ = pso_minimize(inertia_w, bw, n_particles=14, iters=18, seed=3)
    Cw = bestw.reshape(Kw, 2)
    labw = ((Xw[:, None, :] - Cw[None, :, :]) ** 2).sum(axis=2).argmin(1)
    silw = float(silhouette_score(Xw, labw))
    kmw = KMeans(n_clusters=Kw, n_init=10, random_state=SEED).fit(Xw)

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(Xw[:, 0], Xw[:, 1], c=labw, cmap="tab10", s=18)
    ax.scatter(Cw[:, 0], Cw[:, 1], c="k", marker="X", s=90, label="centroides PSO")
    ax.set_xlabel(wine.feature_names[0]); ax.set_ylabel(wine.feature_names[1])
    ax.set_title(f"Wine (2 vars) — PSO clustering  sil={silw:.3f}")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "04_wine_clusters.png", bbox_inches="tight"); plt.close()

    return {
        "blobs_inertia_pso": float(cost),
        "blobs_inertia_kmeans": float(km.inertia_),
        "blobs_sil_pso": sil_pso,
        "blobs_sil_kmeans": sil_km,
        "wine2_sil_pso": silw,
        "wine2_sil_kmeans": float(silhouette_score(Xw, kmw.labels_)),
    }


def main():
    out = {}
    print("FS", flush=True)
    out["feature_selection"] = run_feature_selection()
    print(out["feature_selection"], flush=True)
    print("HP", flush=True)
    out["hyperparams"] = run_hyperparams()
    print(out["hyperparams"], flush=True)
    print("NN", flush=True)
    out["nn"] = run_nn()
    print(out["nn"], flush=True)
    print("CL", flush=True)
    out["clustering"] = run_clustering()
    print(out["clustering"], flush=True)
    (FIG / "results.json").write_text(json.dumps(out, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

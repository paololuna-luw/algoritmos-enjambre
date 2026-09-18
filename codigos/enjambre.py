#!/usr/bin/env python3
"""Primitivas ABC y PSO usadas por los cuatro ejercicios."""
from __future__ import annotations

import numpy as np


def pso_minimize(objective, bounds, n_particles=20, iters=25, w=0.7, c1=1.5, c2=1.5, seed=42):
    """PSO global-best. Minimiza objective(x).

    Ciclo: representacion = posicion continua,
    inicializacion uniforme, aptitud = objective,
    comportamiento = inercia + pbest + gbest,
    evolucion por iteraciones, parada = iters.
    """
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
    """ABC binario. Maximiza objective(bitstring).

    Ciclo: representacion = fuente de alimento binaria,
    inicializacion aleatoria, aptitud = objective,
    comportamiento = empleada / observadora / exploradora,
    evolucion por ciclos, parada = cycles.
    """
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

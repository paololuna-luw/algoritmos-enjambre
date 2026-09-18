#!/usr/bin/env python3
"""Corre los cuatro ejercicios en orden y guarda figures/results.json."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SCRIPTS = [
    ("feature_selection", "01_abc_feature_selection.py"),
    ("hyperparams", "02_pso_hyperparameter_tuning.py"),
    ("nn", "03_pso_entrenar_red.py"),
    ("clustering", "04_pso_clustering.py"),
]


def main():
    out = {}
    for key, name in SCRIPTS:
        print("\n########", name, "########")
        ns = runpy.run_path(str(HERE / name), run_name="not_main")
        out[key] = ns["main"]()
    (FIG / "results.json").write_text(json.dumps(out, indent=2))
    print("\nWROTE", FIG / "results.json")


if __name__ == "__main__":
    main()

"""Two-seed reproducibility check.

Retrains the main model with a second seed and reports ΔMAE on the
same held-out test slice.
"""
from __future__ import annotations
import json
import numpy as np
import subprocess
import sys


def run(seed: int, tag: str):
    metrics_path = f"results/metrics_seed{seed}.json"
    preds_path = f"results/preds_test_seed{seed}.npz"
    model_path = f"models/hgbr_seed{seed}.joblib"
    subprocess.check_call([
        sys.executable, "src/train.py",
        "--data", "results/dataset.npz",
        "--seed", str(seed),
        "--model-out", model_path,
        "--metrics-out", metrics_path,
        "--preds-out", preds_path,
    ])
    with open(metrics_path) as f:
        m = json.load(f)
    return m


def _pick(m):
    L = m["lightgbm"]
    return {
        "test_mae": L["test_mae"],
        "test_rmse": L["test_rmse"],
        "test_spearman": L["test_spearman"],
        "test_middle_decile_calibration": L["test_middle_decile_calibration"],
        "calib_alpha": L["calib_alpha"],
    }


if __name__ == "__main__":
    # Seed 42 is already trained as the main model; run seed 1
    m1 = run(1, "seed1")
    print("Seed 1 test MAE:", m1["lightgbm"]["test_mae"])
    with open("results/metrics.json") as f:
        m42 = json.load(f)
    print("Seed 42 test MAE:", m42["lightgbm"]["test_mae"])
    delta_mae = abs(m1["lightgbm"]["test_mae"] - m42["lightgbm"]["test_mae"])
    delta_spear = abs(m1["lightgbm"]["test_spearman"] - m42["lightgbm"]["test_spearman"])
    print(f"ΔMAE: {delta_mae:.2f}  ΔSpearman: {delta_spear:.4f}")
    with open("results/reproducibility.json", "w") as f:
        json.dump({
            "seed42": _pick(m42),
            "seed1": _pick(m1),
            "delta_mae": delta_mae,
            "delta_spearman": delta_spear,
        }, f, indent=2)

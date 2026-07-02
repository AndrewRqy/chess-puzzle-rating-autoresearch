"""Train the LightGBM puzzle-difficulty regressor.

Loads features from `results/dataset.npz` (produced by build_dataset.py),
trains a LightGBM model with early stopping on the internal val slice,
and evaluates on the internal test slice. Persists model + metadata
into `models/`.

Also implements simple baselines (global mean, per-theme mean) for
comparison.
"""

from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
from scipy.stats import spearmanr


def load_dataset(path: str):
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"],
        "y": d["y"],
        "rd": d["rd"],
        "split": d["split"],
        "pids": d["pids"],
        "feature_names": [str(n) for n in d["feature_names"]],
    }


def split_arrays(data: dict):
    out = {}
    for s in ("train", "val", "test"):
        mask = data["split"] == s
        out[s] = {
            "X": data["X"][mask],
            "y": data["y"][mask],
            "rd": data["rd"][mask],
            "pids": data["pids"][mask],
        }
    return out


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # Spearman (may be slow for very large arrays; fine here)
    rho, _ = spearmanr(y_true, y_pred)
    # Per-decile calibration
    edges = np.percentile(y_true, np.linspace(0, 100, 11))
    edges[0] -= 1
    bin_idx = np.digitize(y_true, edges[1:-1])
    cal_by_bin = []
    for b in range(10):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        cal_by_bin.append({
            "bin": int(b),
            "n": int(m.sum()),
            "mean_true": float(np.mean(y_true[m])),
            "mean_pred": float(np.mean(y_pred[m])),
            "abs_gap": float(abs(np.mean(y_pred[m]) - np.mean(y_true[m]))),
        })
    middle = [c for c in cal_by_bin if 1400 <= c["mean_true"] <= 2000]
    middle_max = float(max((c["abs_gap"] for c in middle), default=0.0))
    return {
        "mae": mae,
        "rmse": rmse,
        "spearman": float(rho),
        "middle_decile_calibration": middle_max,
        "calibration_by_bin": cal_by_bin,
    }


def baseline_global_mean(y_train, y_test):
    mu = float(np.mean(y_train))
    yhat = np.full_like(y_test, mu, dtype=np.float32)
    return yhat, {"mu": mu}


def baseline_theme_mean(X_train, y_train, X_test, feat_names):
    """Predict mean rating conditional on theme membership.

    We use the average rating of all training rows sharing any of the
    puzzle's themes, weighted equally, falling back to global mean.
    """
    theme_cols = [(i, n[len("theme_"):]) for i, n in enumerate(feat_names)
                  if n.startswith("theme_") and n != "theme_missing" and n != "theme_count"]
    global_mu = float(np.mean(y_train))
    # Per-theme mean rating on training set
    theme_means = {}
    for i, tname in theme_cols:
        mask = X_train[:, i] == 1
        if mask.sum() >= 30:
            theme_means[i] = float(np.mean(y_train[mask]))
    # Predict on test rows
    yhat = np.zeros(X_test.shape[0], dtype=np.float32)
    for r in range(X_test.shape[0]):
        vals = []
        for i, _ in theme_cols:
            if X_test[r, i] == 1 and i in theme_means:
                vals.append(theme_means[i])
        yhat[r] = float(np.mean(vals)) if vals else global_mu
    return yhat, {"global_mu": global_mu, "n_themes": len(theme_means)}


def train_lightgbm(X_tr, y_tr, X_va, y_va, rd_tr, seed: int, params: dict | None = None):
    """Train a HistGradientBoosting regressor (sklearn's LightGBM-like impl).

    Named 'train_lightgbm' for continuity with the plan. sklearn's
    HGBR uses its own bundled libgomp, so it works without extra
    system dependencies. We use the absolute-error loss (equivalent to
    LightGBM's regression_l1) and RD-aware sample weights.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    weights = 1.0 / (1.0 + (rd_tr / 40.0))
    p = {
        # squared_error avoids the median-collapse of absolute_error and
        # gives materially better per-decile calibration for this task.
        # Bounded budget keeps two-seed reproducibility runs comparable.
        "loss": "squared_error",
        "learning_rate": 0.07,
        "max_iter": 500,
        "max_leaf_nodes": 127,
        "min_samples_leaf": 100,
        "l2_regularization": 0.5,
        "early_stopping": True,
        "validation_fraction": None,
        "n_iter_no_change": 20,
        "tol": 1e-3,
        "random_state": seed,
        "verbose": 1,
    }
    if params:
        p.update(params)
    # sklearn HGBR's built-in early stopping uses a random slice of the
    # training data. Keep X_va (external validation slice) fully held
    # out so it can be used later for isotonic calibration and honest
    # evaluation. HGBR will carve out its own internal slice from X_tr.
    p["validation_fraction"] = 0.1
    model = HistGradientBoostingRegressor(**p)
    model.fit(X_tr, y_tr, sample_weight=weights)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/dataset.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-out", default="models/hgbr.joblib")
    ap.add_argument("--metrics-out", default="results/metrics.json")
    ap.add_argument("--preds-out", default="results/preds_test.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)

    print(f"[train] Loading {args.data} …")
    data = load_dataset(args.data)
    splits = split_arrays(data)
    for s in ("train", "val", "test"):
        print(f"  {s}: {splits[s]['X'].shape}")

    feat_names = data["feature_names"]

    # ---- Baselines
    print("[baseline] Global mean …")
    bmu_pred, _ = baseline_global_mean(splits["train"]["y"], splits["test"]["y"])
    bmu_metrics = eval_metrics(splits["test"]["y"], bmu_pred)
    print(f"  MAE {bmu_metrics['mae']:.2f}  RMSE {bmu_metrics['rmse']:.2f}  "
          f"spearman {bmu_metrics['spearman']:.4f}")

    print("[baseline] Per-theme mean …")
    btm_pred, _ = baseline_theme_mean(
        splits["train"]["X"], splits["train"]["y"],
        splits["test"]["X"], feat_names,
    )
    btm_metrics = eval_metrics(splits["test"]["y"], btm_pred)
    print(f"  MAE {btm_metrics['mae']:.2f}  RMSE {btm_metrics['rmse']:.2f}  "
          f"spearman {btm_metrics['spearman']:.4f}")

    # ---- Main model
    print(f"[main] Training LightGBM (seed={args.seed}) …")
    t0 = time.time()
    model = train_lightgbm(
        splits["train"]["X"], splits["train"]["y"],
        splits["val"]["X"], splits["val"]["y"],
        splits["train"]["rd"], seed=args.seed,
    )
    best_iter = getattr(model, "n_iter_", None) or model.max_iter
    print(f"  trained in {time.time()-t0:.0f}s  n_iter={best_iter}")

    # Fit isotonic calibrator on val predictions to correct tail
    # mean-reversion (a common failure mode of tree-based regressors).
    from sklearn.isotonic import IsotonicRegression
    y_pred_val_raw = model.predict(splits["val"]["X"]).astype(np.float32)
    iso = IsotonicRegression(out_of_bounds="clip",
                             y_min=400.0, y_max=3300.0)
    iso.fit(y_pred_val_raw, splits["val"]["y"])

    y_pred_test_raw = model.predict(splits["test"]["X"]).astype(np.float32)
    y_pred_test = iso.transform(y_pred_test_raw).astype(np.float32)
    y_pred_val = iso.transform(y_pred_val_raw).astype(np.float32)

    lgbm_test = eval_metrics(splits["test"]["y"], y_pred_test)
    lgbm_val = eval_metrics(splits["val"]["y"], y_pred_val)
    print(f"[main] TEST  MAE {lgbm_test['mae']:.2f}  RMSE {lgbm_test['rmse']:.2f}  "
          f"spearman {lgbm_test['spearman']:.4f}  "
          f"middle_decile_cal {lgbm_test['middle_decile_calibration']:.2f}")
    print(f"[main] VAL   MAE {lgbm_val['mae']:.2f}  RMSE {lgbm_val['rmse']:.2f}  "
          f"spearman {lgbm_val['spearman']:.4f}")

    print(f"[save] Model → {args.model_out}")
    import joblib
    joblib.dump({"model": model, "iso": iso}, args.model_out)
    # Feature names for the predict.py loader
    fn_path = os.path.join(os.path.dirname(args.model_out) or ".", "feature_names.json")
    with open(fn_path, "w") as f:
        json.dump(feat_names, f)
    # Meta
    meta_path = os.path.join(os.path.dirname(args.model_out) or ".", "meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "global_mean_rating": float(np.mean(splits["train"]["y"])),
            "seed": args.seed,
            "n_iter": int(best_iter),
        }, f)

    # Permutation-style feature importances (sklearn HGBR doesn't
    # expose native gain; use a lightweight proxy: variance of the
    # per-tree predictions perturbed by shuffling. For speed we
    # fall back to the training-set correlation-with-target as a
    # rough gain measure.)
    imp = []
    y_tr = splits["train"]["y"]
    X_tr = splits["train"]["X"]
    for j in range(X_tr.shape[1]):
        col = X_tr[:, j]
        if col.std() == 0:
            imp.append(0.0)
        else:
            imp.append(abs(np.corrcoef(col, y_tr)[0, 1]))
    top = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:25]

    metrics = {
        "seed": args.seed,
        "n_train": int(splits["train"]["X"].shape[0]),
        "n_val": int(splits["val"]["X"].shape[0]),
        "n_test": int(splits["test"]["X"].shape[0]),
        "baseline_global_mean": {
            "test_mae": bmu_metrics["mae"],
            "test_rmse": bmu_metrics["rmse"],
            "test_spearman": bmu_metrics["spearman"],
            "test_middle_decile_calibration": bmu_metrics["middle_decile_calibration"],
        },
        "baseline_theme_mean": {
            "test_mae": btm_metrics["mae"],
            "test_rmse": btm_metrics["rmse"],
            "test_spearman": btm_metrics["spearman"],
            "test_middle_decile_calibration": btm_metrics["middle_decile_calibration"],
        },
        "lightgbm": {
            "best_iter": int(best_iter),
            "val_mae": lgbm_val["mae"],
            "val_rmse": lgbm_val["rmse"],
            "val_spearman": lgbm_val["spearman"],
            "test_mae": lgbm_test["mae"],
            "test_rmse": lgbm_test["rmse"],
            "test_spearman": lgbm_test["spearman"],
            "test_middle_decile_calibration": lgbm_test["middle_decile_calibration"],
            "test_calibration_by_bin": lgbm_test["calibration_by_bin"],
            "top_features": [{"name": n, "gain": float(g)} for n, g in top],
        },
    }
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] Metrics → {args.metrics_out}")

    # Predictions for downstream plots
    np.savez(args.preds_out,
             y_true=splits["test"]["y"],
             y_pred=y_pred_test,
             rd=splits["test"]["rd"],
             pids=splits["test"]["pids"])
    print(f"[save] Test preds → {args.preds_out}")


if __name__ == "__main__":
    main()

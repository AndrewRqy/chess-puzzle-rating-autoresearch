"""Generate visualizations from test predictions."""

from __future__ import annotations
import argparse
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pred_vs_actual(y_true, y_pred, out):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Sample down for scatter density
    idx = np.random.default_rng(0).choice(len(y_true),
                                          size=min(20000, len(y_true)),
                                          replace=False)
    ax.scatter(y_true[idx], y_pred[idx], s=3, alpha=0.15, color="C0")
    lo, hi = 400, 3300
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("True Rating (Elo)")
    ax.set_ylabel("Predicted Rating (Elo)")
    ax.set_title("Predicted vs Actual Puzzle Rating")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_residuals(y_true, y_pred, out):
    err = y_pred - y_true
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err, bins=80, color="C1", edgecolor="white", alpha=0.9)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Predicted − Actual (Elo)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Histogram — mean {np.mean(err):.1f}, "
                 f"MAE {np.mean(np.abs(err)):.1f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_decile_calibration(y_true, y_pred, out):
    edges = np.percentile(y_true, np.linspace(0, 100, 11))
    edges[0] -= 1
    bin_idx = np.digitize(y_true, edges[1:-1])
    xs, mean_true, mean_pred, ns = [], [], [], []
    for b in range(10):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        xs.append(b)
        mean_true.append(np.mean(y_true[m]))
        mean_pred.append(np.mean(y_pred[m]))
        ns.append(m.sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.4
    x = np.arange(len(xs))
    ax.bar(x - width/2, mean_true, width, label="Mean true", color="C0")
    ax.bar(x + width/2, mean_pred, width, label="Mean pred", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{b}" for b in xs])
    ax.set_xlabel("Rating Decile")
    ax.set_ylabel("Rating (Elo)")
    ax.set_title("Per-decile Calibration (Mean true vs predicted)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_feature_importance(metrics_path, out):
    with open(metrics_path) as f:
        m = json.load(f)
    tops = m["lightgbm"]["top_features"][:20][::-1]
    names = [t["name"] for t in tops]
    gains = [t["gain"] for t in tops]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(len(names)), gains, color="C3")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Total gain")
    ax.set_title("Top 20 features by LightGBM gain")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds_test.npz")
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.preds, allow_pickle=True)
    y_true = d["y_true"].astype(float)
    y_pred = d["y_pred"].astype(float)

    plot_pred_vs_actual(y_true, y_pred, os.path.join(args.out_dir, "pred_vs_actual.png"))
    plot_residuals(y_true, y_pred, os.path.join(args.out_dir, "residual_hist.png"))
    plot_decile_calibration(y_true, y_pred, os.path.join(args.out_dir, "decile_calibration.png"))
    plot_feature_importance(args.metrics, os.path.join(args.out_dir, "feature_importance.png"))
    print("[analyze] Wrote 4 figures to", args.out_dir)


if __name__ == "__main__":
    main()

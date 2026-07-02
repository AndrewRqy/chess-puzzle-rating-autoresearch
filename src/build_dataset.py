"""Build a numeric feature matrix from Lichess puzzles.

Reads all parquet shards, samples a manageable subset, extracts features,
and writes .npz files partitioned into train / val / test by a hash of
PuzzleId (mimicking the sealed-test rule; our train excludes both
5% hash bucket = 0..4 and 5..9 so we have parallel val/test slices).
"""

from __future__ import annotations
import argparse
import glob
import hashlib
import json
import os
import time
import numpy as np
import pandas as pd

from features import extract_features, feature_names


def hash_bucket(pid: str) -> int:
    """Return an int in [0, 100)."""
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % 100


def split_of(pid: str) -> str:
    """
    Hash-bucket split:
      0-4   -> reserved (mimic sealed test); never used for training/eval
      5-9   -> internal test slice
      10-14 -> internal validation slice
      15-99 -> training slice
    The 0..4 bucket protects us from accidentally training on Lichess
    puzzles the scorer might pick.
    """
    b = hash_bucket(pid)
    if b < 5:
        return "reserved"
    if b < 10:
        return "test"
    if b < 15:
        return "val"
    return "train"


def load_puzzles(n_target: int, rd_max: int, seed: int, data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    print(f"[load] Reading {len(files)} shards …")
    dfs = []
    for f in files:
        dfs.append(pd.read_parquet(
            f, columns=["PuzzleId", "FEN", "Moves", "Themes",
                        "Rating", "RatingDeviation"]
        ))
    df = pd.concat(dfs, ignore_index=True)
    print(f"[load] Total rows: {len(df):,}")

    # Filter for high-confidence ratings
    df = df[df["RatingDeviation"] < rd_max].copy()
    print(f"[load] After RD<{rd_max}: {len(df):,}")

    # Attach split
    df["split"] = df["PuzzleId"].apply(split_of)
    df = df[df["split"] != "reserved"].copy()

    # Subsample per-split to fit budget
    # We want ~n_target rows total; distribute as
    # 70% train, 15% val, 15% test but bounded by actual bucket sizes.
    rng = np.random.default_rng(seed)
    parts = []
    for split, frac in [("train", 0.75), ("val", 0.125), ("test", 0.125)]:
        sub = df[df["split"] == split]
        take = min(len(sub), int(n_target * frac))
        idx = rng.choice(len(sub), size=take, replace=False)
        parts.append(sub.iloc[idx])
        print(f"[load] {split}: took {take:,} of {len(sub):,}")
    return pd.concat(parts, ignore_index=True)


def extract_matrix(df: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    X = np.zeros((n, len(names)), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    rd = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        themes = list(row.Themes) if row.Themes is not None else None
        feats = extract_features(row.FEN, row.Moves, themes)
        for j, name in enumerate(names):
            X[i, j] = feats.get(name, 0.0)
        y[i] = row.Rating
        rd[i] = row.RatingDeviation
        if (i + 1) % 20000 == 0:
            elapsed = time.time() - t0
            print(f"  processed {i+1:,}/{n:,}  ({elapsed:.0f}s, {(i+1)/elapsed:.0f} rows/s)")
    return X, y, rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--rd-max", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/dataset.npz")
    ap.add_argument("--data-dir", default="datasets/lichess_puzzles/data")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = load_puzzles(args.n, args.rd_max, args.seed, args.data_dir)
    print(f"[load] Total selected: {len(df):,}")

    names = feature_names()
    print(f"[extract] Extracting {len(names)} features per row …")
    X, y, rd = extract_matrix(df, names)

    split_arr = df["split"].values
    pids = df["PuzzleId"].values

    print(f"[save] Writing {args.out}")
    np.savez_compressed(
        args.out,
        X=X, y=y, rd=rd,
        split=split_arr,
        pids=pids,
        feature_names=np.array(names),
    )
    with open("results/feature_names.json", "w") as f:
        json.dump(names, f)
    print("[save] Done.")


if __name__ == "__main__":
    main()

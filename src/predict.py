"""Sealed-scoring entry point.

Exposes `predict(rows: list[dict]) -> list[float]` for the post-run
scorer. Loads a LightGBM model persisted at `models/lgbm.txt` and a
frozen feature-name list at `models/feature_names.json`.

- No network, no disk writes.
- Deterministic: same rows in → same floats out.
- Handles missing / empty `Themes` gracefully.
"""

from __future__ import annotations
import json
import os
from typing import Any

import numpy as np

_MODEL = None
_FEATURE_NAMES: list[str] | None = None
_GLOBAL_MEAN: float | None = None


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _load_artifacts() -> None:
    """Lazily load the boosted-tree regressor + feature-name list."""
    global _MODEL, _FEATURE_NAMES, _GLOBAL_MEAN
    if _MODEL is not None:
        return
    import joblib
    root = _here()
    model_path = os.path.join(root, "..", "models", "hgbr.joblib")
    names_path = os.path.join(root, "..", "models", "feature_names.json")
    meta_path = os.path.join(root, "..", "models", "meta.json")
    _MODEL = joblib.load(model_path)  # dict with keys 'model', 'iso'
    with open(names_path) as f:
        _FEATURE_NAMES = json.load(f)
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        _GLOBAL_MEAN = float(meta.get("global_mean_rating", 1500.0))
    except Exception:
        _GLOBAL_MEAN = 1500.0


def _row_to_features(row: dict[str, Any]) -> dict[str, float]:
    """Extract features for a single row (safe wrapper for the scorer)."""
    # Import here so that if src/features.py is unavailable we still error early
    from features import extract_features
    fen = row.get("FEN", "")
    moves = row.get("Moves", "") or ""
    themes = row.get("Themes")
    if themes is not None:
        try:
            themes = list(themes)
        except TypeError:
            themes = None
    try:
        return extract_features(fen, moves, themes)
    except Exception:
        # Return an empty dict; downstream will fill with zeros
        return {}


def predict(rows: list[dict]) -> list[float]:
    """Predict Lichess Elo rating for each puzzle row.

    Robust to bad rows: any per-row failure falls back to a global-mean
    prediction so that the returned list length always matches `rows`.
    """
    _load_artifacts()
    assert _MODEL is not None and _FEATURE_NAMES is not None

    n = len(rows)
    if n == 0:
        return []

    X = np.zeros((n, len(_FEATURE_NAMES)), dtype=np.float32)
    bad_rows: list[int] = []
    for i, r in enumerate(rows):
        feats = _row_to_features(r)
        if not feats:
            bad_rows.append(i)
            continue
        for j, name in enumerate(_FEATURE_NAMES):
            v = feats.get(name, 0.0)
            if v is None:
                v = 0.0
            X[i, j] = float(v)

    raw = _MODEL["model"].predict(X).astype(np.float32)
    iso_p = _MODEL["iso"].transform(raw).astype(np.float32)
    pred_sorted = _MODEL["calib_pred_sorted"]
    true_sorted = _MODEL["calib_true_sorted"]
    alpha = float(_MODEL["calib_alpha"])
    ranks = np.searchsorted(pred_sorted, raw, side="left") / float(
        len(pred_sorted)
    )
    cdf_p = np.interp(
        ranks,
        np.linspace(0.0, 1.0, len(true_sorted), dtype=np.float32),
        true_sorted,
    ).astype(np.float32)
    cdf_p = np.clip(cdf_p, float(true_sorted[0]), float(true_sorted[-1]))
    preds = alpha * iso_p + (1.0 - alpha) * cdf_p
    preds = np.asarray(preds, dtype=np.float64)
    # Clip to plausible Elo range so no NaN/Inf can escape
    preds = np.clip(preds, 400.0, 3300.0)
    if bad_rows:
        for i in bad_rows:
            preds[i] = _GLOBAL_MEAN if _GLOBAL_MEAN is not None else 1500.0
    # Final safety scrub
    preds = np.nan_to_num(preds, nan=1500.0, posinf=3300.0, neginf=400.0)
    return preds.astype(float).tolist()


# Import-time ensure features module reachable via `import features`
# by inserting the src/ dir onto sys.path.
def _ensure_local_import() -> None:
    import sys
    root = _here()
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_local_import()

"""Explainable AI layer.

Two independent, honestly-computed views:

* GLOBAL  - permutation feature importance measured on the held-out test set
            (computed in modeling.py) + global mean |SHAP| over a data sample.
* LOCAL   - EXACT Shapley values for the operating point the engineer entered.

Because the surrogate has only 4 features, the Shapley value can be computed
exhaustively over all 2^4 = 16 coalitions with an interventional (marginal)
value function estimated on a background sample drawn from the TCAD DOE data.
That is an exact SHAP computation - no sampling approximation, no external
dependency - and it satisfies the efficiency axiom:

    sum(phi_j) = f(x) - E[f(X)]

which the UI displays as a check.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Any, Callable, Sequence

import numpy as np

from .config import FEATURE_COLUMNS, FEATURE_LABELS, FEATURE_LABELS_KO, MODEL_TARGETS
from .dataset import (
    build_features,
    features_from_arrays,
    frame,
    input_bounds,
    load_dataset,
)


def _predictor(model: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Predict callable that always feeds named columns to the estimator."""
    return lambda Z: np.asarray(model.predict(frame(Z)), dtype=float)

N_FEATURES = len(FEATURE_COLUMNS)
_SUBSETS: list[tuple[int, ...]] = []
for _r in range(N_FEATURES + 1):
    _SUBSETS.extend(combinations(range(N_FEATURES), _r))
_SUBSET_INDEX = {s: i for i, s in enumerate(_SUBSETS)}


def _shapley_weight(s_size: int, n: int) -> float:
    return factorial(s_size) * factorial(n - s_size - 1) / factorial(n)


def exact_shap_values(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Exact interventional Shapley values for a single instance.

    Returns (phi[n_features], base_value, f(x)).
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    bg = np.asarray(background, dtype=float)
    n_bg = bg.shape[0]

    # Build every coalition's counterfactual batch in one matrix -> one predict.
    batch = np.repeat(bg[None, :, :], len(_SUBSETS), axis=0)  # (n_subsets, n_bg, n_feat)
    for si, subset in enumerate(_SUBSETS):
        for j in subset:
            batch[si, :, j] = x[j]
    flat = batch.reshape(-1, N_FEATURES)
    preds = np.asarray(predict_fn(flat), dtype=float).reshape(len(_SUBSETS), n_bg)
    v = preds.mean(axis=1)  # value function per coalition

    phi = np.zeros(N_FEATURES)
    for j in range(N_FEATURES):
        others = [k for k in range(N_FEATURES) if k != j]
        for r in range(len(others) + 1):
            for s in combinations(others, r):
                s_sorted = tuple(sorted(s))
                s_with_j = tuple(sorted(s + (j,)))
                phi[j] += _shapley_weight(len(s), N_FEATURES) * (
                    v[_SUBSET_INDEX[s_with_j]] - v[_SUBSET_INDEX[s_sorted]]
                )
    base = float(v[_SUBSET_INDEX[()]])
    fx = float(v[_SUBSET_INDEX[tuple(range(N_FEATURES))]])
    return phi, base, fx


def background_sample(n: int = 160, seed: int = 7) -> np.ndarray:
    """Representative background drawn from the real TCAD DOE data."""
    df = load_dataset()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    return build_features(df.iloc[idx]).to_numpy(dtype=float)


def global_shap_importance(
    models: dict[str, Any], n_sample: int = 120, n_background: int = 120, seed: int = 11
) -> dict[str, dict[str, float]]:
    """Mean |SHAP| per feature over a random sample of the DOE dataset."""
    df = load_dataset()
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    X_sample = build_features(df.iloc[sample_idx]).to_numpy(dtype=float)
    bg = background_sample(n_background, seed=seed + 1)

    out: dict[str, dict[str, float]] = {}
    for target, model in models.items():
        acc = np.zeros(N_FEATURES)
        for row in X_sample:
            phi, _, _ = exact_shap_values(_predictor(model), row, bg)
            acc += np.abs(phi)
        acc /= len(X_sample)
        out[target] = {FEATURE_COLUMNS[i]: float(acc[i]) for i in range(N_FEATURES)}
    return out


def _rank_payload(
    values: dict[str, float], signed: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    total = sum(abs(v) for v in values.values()) or 1.0
    items = []
    for key, val in values.items():
        items.append(
            {
                "feature": key,
                "label": FEATURE_LABELS[key],
                "label_ko": FEATURE_LABELS_KO[key],
                "value": float(val),
                "abs_value": float(abs(val)),
                "share_pct": float(abs(val) / total * 100.0),
                "signed": float(signed[key]) if signed else float(val),
            }
        )
    items.sort(key=lambda d: -d["abs_value"])
    for rank, item in enumerate(items, start=1):
        item["rank"] = rank
    return items


def local_explanation(
    dose: float,
    energy: float,
    temp: float,
    time_s: float,
    targets: Sequence[str] = tuple(MODEL_TARGETS),
    n_background: int = 160,
) -> dict[str, Any]:
    """Per-target exact SHAP attribution + local one-at-a-time sensitivity."""
    from .surrogate import load_bundle  # local import avoids circular import

    bundle = load_bundle()
    bg = background_sample(n_background)
    x = features_from_arrays([dose], [energy], [temp], [time_s])[0]
    bounds = input_bounds()
    raw_values = {
        "log10_dose": float(dose),
        "energy_keV": float(energy),
        "anneal_temp_C": float(temp),
        "anneal_time_sec": float(time_s),
    }

    out: dict[str, Any] = {}
    for target in targets:
        model = bundle["models"][target]
        predict_fn = _predictor(model)
        phi, base, fx = exact_shap_values(predict_fn, x, bg)
        contrib = {FEATURE_COLUMNS[i]: float(phi[i]) for i in range(N_FEATURES)}

        # Local sensitivity: sweep each parameter across its full DOE range
        # while the other three stay at the engineer's operating point.
        sens: dict[str, float] = {}
        sens_dir: dict[str, float] = {}
        for i, feat in enumerate(FEATURE_COLUMNS):
            src = {
                "log10_dose": "dose_cm2",
                "energy_keV": "energy_keV",
                "anneal_temp_C": "anneal_temp_C",
                "anneal_time_sec": "anneal_time_sec",
            }[feat]
            lo, hi = bounds[src]["min"], bounds[src]["max"]
            if feat == "log10_dose":
                lo, hi = float(np.log10(lo)), float(np.log10(hi))
            grid = np.linspace(lo, hi, 21)
            batch = np.repeat(x[None, :], len(grid), axis=0)
            batch[:, i] = grid
            preds = predict_fn(batch)
            sens[feat] = float(preds.max() - preds.min())
            sens_dir[feat] = float(preds[-1] - preds[0])

        out[target] = {
            "prediction": fx,
            "base_value": base,
            "efficiency_residual": float(fx - base - phi.sum()),
            "shap": _rank_payload(contrib),
            "sensitivity": _rank_payload(sens, signed=sens_dir),
            "method": "Exact interventional Shapley (2^4 coalitions, TCAD-data background)",
        }
    out["_operating_point"] = raw_values
    return out


def global_explanation() -> dict[str, Any]:
    """Global XAI payload assembled from the stored training report."""
    from .surrogate import load_bundle

    report = load_bundle()["report"]
    out: dict[str, Any] = {"targets": {}}
    for target in MODEL_TARGETS:
        t = report["targets"][target]
        perm = {k: v["mean"] for k, v in t["global_importance"].items()}
        perm_std = {k: v["std"] for k, v in t["global_importance"].items()}
        shap_imp = report.get("global_shap", {}).get(target, {})
        out["targets"][target] = {
            "best_model": t["best_model"],
            "permutation": _rank_payload(perm),
            "permutation_std": perm_std,
            "shap_mean_abs": _rank_payload(shap_imp) if shap_imp else [],
        }
    out["method"] = {
        "permutation": (
            "sklearn.inspection.permutation_importance, 30 repeats, scoring=R2, "
            "evaluated on the held-out test split"
        ),
        "shap": (
            "Exact interventional Shapley values averaged over a random DOE sample "
            "(mean |phi| per process variable)"
        ),
    }
    return out

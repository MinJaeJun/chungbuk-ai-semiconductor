"""Runtime surrogate: loads the trained bundle and serves fast predictions."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    ARTIFACT_PATH,
    FEATURE_COLUMNS,
    MODEL_TARGETS,
    PROCESS_INPUTS,
)
from .dataset import features_from_arrays, frame, input_bounds, load_dataset

_BUNDLE: dict[str, Any] | None = None


class SurrogateNotTrained(RuntimeError):
    pass


def load_bundle(force: bool = False) -> dict[str, Any]:
    global _BUNDLE
    if _BUNDLE is None or force:
        if not ARTIFACT_PATH.exists():
            raise SurrogateNotTrained(
                f"Model artifact not found: {ARTIFACT_PATH}\n"
                "Run:  python train_model.py"
            )
        _BUNDLE = joblib.load(ARTIFACT_PATH)
    return _BUNDLE


def is_trained() -> bool:
    return ARTIFACT_PATH.exists()


def _as_array(value: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float))


# Kernel-based models (Gaussian Process) build an (n_query x n_train) matrix
# on every call, so large optimizer grids are predicted in bounded chunks.
PREDICT_CHUNK = 8192


def _predict_chunked(model: Any, X: pd.DataFrame) -> np.ndarray:
    n = len(X)
    if n <= PREDICT_CHUNK:
        return np.asarray(model.predict(X), dtype=float)
    out = np.empty(n, dtype=float)
    for start in range(0, n, PREDICT_CHUNK):
        stop = min(start + PREDICT_CHUNK, n)
        out[start:stop] = np.asarray(model.predict(X.iloc[start:stop]), dtype=float)
    return out


def predict_raw(
    dose: Any, energy: Any, temp: Any, time_s: Any
) -> dict[str, np.ndarray]:
    """Vectorised surrogate prediction for all targets + derived delta_xj."""
    bundle = load_bundle()
    X = frame(
        features_from_arrays(
            _as_array(dose), _as_array(energy), _as_array(temp), _as_array(time_s)
        )
    )
    out: dict[str, np.ndarray] = {}
    for target in MODEL_TARGETS:
        out[target] = _predict_chunked(bundle["models"][target], X)
    # delta is DERIVED, never modelled directly (dataset identity holds exactly)
    out["delta_xj_um"] = out["xj_final_um"] - out["xj_implant_um"]
    return out


def model_uncertainty(
    dose: Any, energy: Any, temp: Any, time_s: Any
) -> dict[str, dict[str, Any]]:
    """Per-target 1-sigma MODEL uncertainty at the queried point.

    This is epistemic uncertainty - how unsure the surrogate is about its own
    answer - and is a different quantity from the equipment scatter handled in
    core/robust.py. A Gaussian Process exposes it natively; a tree ensemble
    approximates it with the spread across trees. When neither applies the
    held-out test RMSE is reported as a flat fallback, so the UI never claims a
    precision the model has not demonstrated.
    """
    bundle = load_bundle()
    X = frame(
        features_from_arrays(
            _as_array(dose), _as_array(energy), _as_array(temp), _as_array(time_s)
        )
    )
    metrics = bundle["report"]["targets"]
    out: dict[str, dict[str, Any]] = {}
    for target in MODEL_TARGETS:
        model = bundle["models"][target]
        rmse = float(metrics[target]["metrics"]["test"]["rmse"])
        std: np.ndarray | None = None
        source = "held-out test RMSE (flat fallback)"

        estimator = model[-1] if hasattr(model, "steps") else model
        try:
            if hasattr(estimator, "kernel_"):  # Gaussian Process
                Xt = model[:-1].transform(X) if hasattr(model, "steps") else X
                _, std = estimator.predict(Xt, return_std=True)
                source = "Gaussian Process posterior std"
            elif hasattr(estimator, "estimators_"):  # tree ensemble
                Xt = model[:-1].transform(X) if hasattr(model, "steps") else X
                arr_t = np.asarray(Xt)
                preds = np.column_stack([t.predict(arr_t) for t in estimator.estimators_])
                std = preds.std(axis=1, ddof=1)
                source = "tree-ensemble spread (std across estimators)"
        except Exception:
            std = None

        arr = np.full(len(X), rmse) if std is None else np.asarray(std, dtype=float)
        out[target] = {
            "std": [float(v) for v in arr],
            "source": source,
            "test_rmse": rmse,
        }
    return out


def delta_resolution() -> dict[str, Any]:
    """Noise floor of the DERIVED delta_xj.

    delta_xj is the difference of two separately predicted junction depths, so
    its resolution is bounded by the combined hold-out error of both models.
    Anything smaller is below the surrogate's resolving power and is reported
    as such instead of being presented as a meaningful number.
    """
    report = load_bundle()["report"]
    m = report["targets"]
    combined_rmse = float(
        np.hypot(
            m["xj_final_um"]["metrics"]["test"]["rmse"],
            m["xj_implant_um"]["metrics"]["test"]["rmse"],
        )
    )
    derived = report.get("derived_delta", {}).get("test", {})
    return {
        "combined_rmse": combined_rmse,
        "measured_rmse": derived.get("rmse"),
        "measured_mae": derived.get("mae"),
        "measured_r2": derived.get("r2"),
        "resolution_limit": float(derived.get("rmse") or combined_rmse),
        "note": (
            "delta_xj is derived as (predicted xj_final - predicted xj_implant). "
            "Values below the resolution limit are within surrogate noise."
        ),
    }


def extrapolation_report(
    dose: float, energy: float, temp: float, time_s: float
) -> dict[str, Any]:
    """Flag any input outside the TCAD training envelope, plus grid distance."""
    bounds = input_bounds()
    df = load_dataset()
    values = {
        "dose_cm2": float(dose),
        "energy_keV": float(energy),
        "anneal_temp_C": float(temp),
        "anneal_time_sec": float(time_s),
    }
    details = []
    outside = []
    for name in PROCESS_INPUTS:
        lo, hi = bounds[name]["min"], bounds[name]["max"]
        v = values[name]
        span = hi - lo
        if v < lo:
            status, dev = "below_range", (lo - v) / span * 100.0
        elif v > hi:
            status, dev = "above_range", (v - hi) / span * 100.0
        else:
            status, dev = "in_range", 0.0
        levels = np.sort(df[name].unique().astype(float))
        on_grid = bool(np.any(np.isclose(levels, v, rtol=1e-9, atol=1e-9)))
        details.append(
            {
                "parameter": name,
                "value": v,
                "min": float(lo),
                "max": float(hi),
                "status": status,
                "deviation_pct_of_range": float(dev),
                "on_doe_grid": on_grid,
            }
        )
        if status != "in_range":
            outside.append(name)

    on_grid_all = all(d["on_doe_grid"] for d in details)
    if outside:
        level = "extrapolation"
    elif on_grid_all:
        level = "validated_doe_point"
    else:
        level = "interpolation"
    return {
        "level": level,
        "is_extrapolation": bool(outside),
        "outside_parameters": outside,
        "details": details,
        "message_en": {
            "validated_doe_point": "Input matches an existing TCAD DOE point. Prediction is inside validated data.",
            "interpolation": "Input lies inside the TCAD DOE envelope but between simulated grid points (AI interpolation).",
            "extrapolation": "EXTRAPOLATION WARNING - input is outside the TCAD training range. Prediction confidence is NOT reliable.",
        }[level],
        "message_ko": {
            "validated_doe_point": "입력 조건이 실제 TCAD DOE 격자점과 일치합니다. 학습 데이터 내부의 검증된 영역입니다.",
            "interpolation": "입력 조건이 학습 범위 내부이지만 시뮬레이션 격자 사이의 보간 영역입니다. TCAD/Fab 검증을 권장합니다.",
            "extrapolation": "외삽 경고 - 입력 조건이 TCAD 학습 범위를 벗어났습니다. 예측 신뢰도를 보장할 수 없습니다.",
        }[level],
    }


def nearest_doe_run(
    dose: float, energy: float, temp: float, time_s: float
) -> dict[str, Any]:
    """Closest actually-simulated TCAD run (normalised distance in DOE space)."""
    df = load_dataset()
    bounds = input_bounds(df)
    query = {
        "dose_cm2": np.log10(float(dose)),
        "energy_keV": float(energy),
        "anneal_temp_C": float(temp),
        "anneal_time_sec": float(time_s),
    }
    dist = np.zeros(len(df))
    for name in PROCESS_INPUTS:
        col = df[name].to_numpy(dtype=float)
        lo, hi = bounds[name]["min"], bounds[name]["max"]
        if name == "dose_cm2":
            col = np.log10(col)
            lo, hi = np.log10(lo), np.log10(hi)
        span = hi - lo if hi > lo else 1.0
        dist += ((col - query[name]) / span) ** 2
    dist = np.sqrt(dist)
    i = int(np.argmin(dist))
    row = df.iloc[i]
    return {
        "distance": float(dist[i]),
        "exact_match": bool(dist[i] < 1e-9),
        "run": {c: float(row[c]) for c in df.columns},
    }


def model_info() -> dict[str, Any]:
    bundle = load_bundle()
    return {
        "trained_at": bundle["trained_at"],
        "feature_columns": FEATURE_COLUMNS,
        "targets": {t: bundle["report"]["targets"][t]["best_model"] for t in MODEL_TARGETS},
        "metrics": {t: bundle["report"]["targets"][t]["metrics"] for t in MODEL_TARGETS},
    }

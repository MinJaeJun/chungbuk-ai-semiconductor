"""What-If / Process-Window analysis: single-parameter sweeps on the surrogate.

When the three non-swept parameters happen to sit exactly on TCAD DOE levels,
the matching real simulated runs are returned as an overlay so the engineer can
see the surrogate curve against actual TCAD points.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import INPUT_META, MODEL_TARGETS
from .dataset import input_bounds, load_dataset
from .surrogate import extrapolation_report, predict_raw

PARAM_ORDER = ["dose_cm2", "energy_keV", "anneal_temp_C", "anneal_time_sec"]


def sweep(
    parameter: str,
    dose: float,
    energy: float,
    temp: float,
    time_s: float,
    n_points: int = 61,
    lo: float | None = None,
    hi: float | None = None,
) -> dict[str, Any]:
    if parameter not in PARAM_ORDER:
        raise ValueError(f"parameter must be one of {PARAM_ORDER}")

    bounds = input_bounds()
    base = {
        "dose_cm2": float(dose),
        "energy_keV": float(energy),
        "anneal_temp_C": float(temp),
        "anneal_time_sec": float(time_s),
    }
    lo = float(lo) if lo is not None else bounds[parameter]["min"]
    hi = float(hi) if hi is not None else bounds[parameter]["max"]
    n_points = int(max(5, min(n_points, 401)))

    if parameter == "dose_cm2":
        grid = np.logspace(np.log10(lo), np.log10(hi), n_points)
    else:
        grid = np.linspace(lo, hi, n_points)

    args = {k: np.full(n_points, v) for k, v in base.items()}
    args[parameter] = grid
    preds = predict_raw(
        args["dose_cm2"], args["energy_keV"], args["anneal_temp_C"], args["anneal_time_sec"]
    )

    # Real TCAD runs with the other three parameters fixed on DOE levels.
    df = load_dataset()
    mask = np.ones(len(df), dtype=bool)
    for name in PARAM_ORDER:
        if name == parameter:
            continue
        mask &= np.isclose(df[name].to_numpy(dtype=float), base[name], rtol=1e-6, atol=1e-9)
    ref = df[mask].sort_values(parameter)
    reference = {
        "available": bool(len(ref) > 0),
        "n_points": int(len(ref)),
        "x": [float(v) for v in ref[parameter].to_numpy()],
        **{t: [float(v) for v in ref[t].to_numpy()] for t in MODEL_TARGETS + ["delta_xj_um"]},
    }

    curves = {t: [float(v) for v in preds[t]] for t in MODEL_TARGETS + ["delta_xj_um"]}
    deltas = {
        t: {
            "min": float(np.min(preds[t])),
            "max": float(np.max(preds[t])),
            "span": float(np.max(preds[t]) - np.min(preds[t])),
            "at_start": float(preds[t][0]),
            "at_end": float(preds[t][-1]),
            "direction": "increasing"
            if preds[t][-1] > preds[t][0]
            else ("decreasing" if preds[t][-1] < preds[t][0] else "flat"),
        }
        for t in MODEL_TARGETS + ["delta_xj_um"]
    }

    return {
        "parameter": parameter,
        "parameter_label": INPUT_META[parameter]["label"],
        "parameter_label_ko": INPUT_META[parameter]["label_ko"],
        "unit": INPUT_META[parameter]["unit"],
        "base_condition": base,
        "x": [float(v) for v in grid],
        "curves": curves,
        "summary": deltas,
        "doe_reference": reference,
        "extrapolation": extrapolation_report(
            base["dose_cm2"], base["energy_keV"], base["anneal_temp_C"], base["anneal_time_sec"]
        ),
        "range": {"min": float(lo), "max": float(hi), "n_points": n_points},
    }


def sweep_all(
    dose: float, energy: float, temp: float, time_s: float, n_points: int = 41
) -> dict[str, Any]:
    return {p: sweep(p, dose, energy, temp, time_s, n_points=n_points) for p in PARAM_ORDER}

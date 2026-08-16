"""TCAD DOE dataset loading, validation and descriptive statistics.

Ground truth for the whole platform is data/implant_anneal_1000.csv.
Nothing in this module fabricates values - every number returned is computed
from that CSV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CSV_PATH,
    FEATURE_COLUMNS,
    FORBIDDEN_AS_FEATURE,
    INPUT_META,
    MODEL_TARGETS,
    PROCESS_INPUTS,
    TARGET_META,
)

_CACHE: dict[str, Any] = {}


def load_dataset(force: bool = False) -> pd.DataFrame:
    """Read the TCAD DOE CSV (cached)."""
    if force or "df" not in _CACHE:
        if not CSV_PATH.exists():
            raise FileNotFoundError(
                f"TCAD dataset not found: {CSV_PATH}. "
                "Copy implant_anneal_1000.csv into the data/ directory."
            )
        df = pd.read_csv(CSV_PATH)
        missing = [c for c in PROCESS_INPUTS + MODEL_TARGETS if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        _CACHE["df"] = df
    return _CACHE["df"]


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Map raw process parameters -> model feature matrix.

    Leakage guard: this function only ever touches PROCESS_INPUTS.
    """
    leaked = [c for c in frame.columns if c in FORBIDDEN_AS_FEATURE and c in PROCESS_INPUTS]
    if leaked:  # defensive, PROCESS_INPUTS and FORBIDDEN_AS_FEATURE are disjoint
        raise ValueError(f"Data leakage detected: {leaked}")
    out = pd.DataFrame(index=frame.index)
    out["log10_dose"] = np.log10(frame["dose_cm2"].to_numpy(dtype=float))
    out["energy_keV"] = frame["energy_keV"].to_numpy(dtype=float)
    out["anneal_temp_C"] = frame["anneal_temp_C"].to_numpy(dtype=float)
    out["anneal_time_sec"] = frame["anneal_time_sec"].to_numpy(dtype=float)
    return out[FEATURE_COLUMNS]


def features_from_arrays(
    dose: np.ndarray, energy: np.ndarray, temp: np.ndarray, time: np.ndarray
) -> np.ndarray:
    """Fast path for optimizer / sweep grids (no DataFrame overhead)."""
    return np.column_stack(
        [
            np.log10(np.asarray(dose, dtype=float)),
            np.asarray(energy, dtype=float),
            np.asarray(temp, dtype=float),
            np.asarray(time, dtype=float),
        ]
    )

def frame(X: np.ndarray) -> pd.DataFrame:
    """Wrap a raw feature matrix in the column names the models were fit with.

    Estimators fitted on a named DataFrame warn when handed a bare ndarray, so
    every runtime predict path goes through this helper.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return pd.DataFrame(arr, columns=FEATURE_COLUMNS)


def doe_levels(df: pd.DataFrame | None = None) -> dict[str, list[float]]:
    df = load_dataset() if df is None else df
    return {c: sorted(float(v) for v in df[c].unique()) for c in PROCESS_INPUTS}


def input_bounds(df: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
    df = load_dataset() if df is None else df
    return {
        c: {"min": float(df[c].min()), "max": float(df[c].max())}
        for c in PROCESS_INPUTS
    }


def target_bounds(df: pd.DataFrame | None = None) -> dict[str, dict[str, float]]:
    df = load_dataset() if df is None else df
    cols = MODEL_TARGETS + ["delta_xj_um"]
    return {
        c: {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "mean": float(df[c].mean()),
            "std": float(df[c].std()),
        }
        for c in cols
        if c in df.columns
    }


def validate_dataset(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Full data-integrity report used by the Data Explorer tab."""
    df = load_dataset() if df is None else df
    levels = doe_levels(df)
    expected_full_factorial = int(np.prod([len(v) for v in levels.values()]))

    delta_residual = (df["xj_final_um"] - df["xj_implant_um"]) - df["delta_xj_um"]
    combo_counts = df.groupby(PROCESS_INPUTS, sort=False).size()

    columns = []
    for c in df.columns:
        role = "identifier"
        if c in PROCESS_INPUTS:
            role = "process_input"
        elif c in MODEL_TARGETS:
            role = "model_target"
        elif c == "delta_xj_um":
            role = "derived_target"
        meta = INPUT_META.get(c) or TARGET_META.get(c) or {}
        columns.append(
            {
                "name": c,
                "dtype": str(df[c].dtype),
                "role": role,
                "unit": meta.get("unit", "-"),
                "label": meta.get("label", c),
                "label_ko": meta.get("label_ko", c),
                "n_unique": int(df[c].nunique()),
                "n_missing": int(df[c].isna().sum()),
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "mean": float(df[c].mean()),
                "std": float(df[c].std()),
            }
        )

    return {
        "csv_path": str(CSV_PATH),
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_inputs": len(PROCESS_INPUTS),
        "n_model_targets": len(MODEL_TARGETS),
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_conditions": int((combo_counts > 1).sum()),
        "columns": columns,
        "doe": {
            "levels": levels,
            "level_counts": {k: len(v) for k, v in levels.items()},
            "expected_full_factorial": expected_full_factorial,
            "is_full_factorial": bool(
                expected_full_factorial == len(df) and (combo_counts == 1).all()
            ),
            "structure": " x ".join(str(len(v)) for v in levels.values())
            + f" = {expected_full_factorial}",
        },
        "delta_identity": {
            "expression": "delta_xj_um = xj_final_um - xj_implant_um",
            "max_abs_residual": float(delta_residual.abs().max()),
            "holds": bool(delta_residual.abs().max() < 1e-9),
        },
        "input_bounds": input_bounds(df),
        "target_bounds": target_bounds(df),
        "leakage_policy": {
            "features_used": PROCESS_INPUTS,
            "excluded_from_features": FORBIDDEN_AS_FEATURE,
            "note": (
                "Only pre-simulation controllable process parameters are used as "
                "model inputs. No output column and no run_id ever enters the "
                "feature matrix."
            ),
        },
    }


def correlation_matrix(df: pd.DataFrame | None = None) -> dict[str, Any]:
    df = load_dataset() if df is None else df
    cols = PROCESS_INPUTS + MODEL_TARGETS + ["delta_xj_um"]
    sub = df[cols].copy()
    sub["dose_cm2"] = np.log10(sub["dose_cm2"])
    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    labels = [
        "log10(Dose)" if c == "dose_cm2" else TARGET_META.get(c, INPUT_META.get(c, {})).get("label", c)
        for c in cols
    ]
    return {
        "columns": cols,
        "labels": labels,
        "pearson": [[float(v) for v in row] for row in pearson.to_numpy()],
        "spearman": [[float(v) for v in row] for row in spearman.to_numpy()],
    }


def marginal_effects(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Mean of every output at each DOE level of every input.

    Because the design is full factorial and perfectly balanced, these
    marginal means are unconfounded main effects - directly usable as
    'DATA OBSERVATION' statements.
    """
    df = load_dataset() if df is None else df
    out: dict[str, Any] = {}
    for inp in PROCESS_INPUTS:
        grouped = df.groupby(inp)[MODEL_TARGETS + ["delta_xj_um"]].mean()
        out[inp] = {
            "levels": [float(v) for v in grouped.index],
            **{
                t: [float(v) for v in grouped[t].to_numpy()]
                for t in MODEL_TARGETS + ["delta_xj_um"]
            },
        }
    return out


def scatter_payload(df: pd.DataFrame | None = None) -> dict[str, list[float]]:
    """Raw arrays for the front-end interactive plots (1000 points, ~70 KB)."""
    df = load_dataset() if df is None else df
    cols = PROCESS_INPUTS + MODEL_TARGETS + ["delta_xj_um"]
    return {c: [float(v) for v in df[c].to_numpy()] for c in cols}

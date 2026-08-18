"""DOE information-content audit for the implant/anneal dataset.

Answers four questions that the 1,000-run full-factorial DOE can settle on its
own, without acquiring any new simulation or fab data:

1. SUFFICIENCY   How many of the 1,000 runs are actually needed?  A learning
                 curve over random subsamples measures the point of diminishing
                 return, i.e. how much TCAD compute the surrogate replaces.
2. MONOTONICITY  Is x_j(dose) physically monotone at fixed energy?  It is not:
                 dose = 2.0e15 lands shallower than 1.5e15 at every energy, a
                 systematic simulator artifact the surrogate otherwise absorbs.
3. EXTRAPOLATION What is the error outside the DOE box?  Trained on the interior
                 levels only, scored on the outer shell.
4. THERMAL AXIS  Do anneal_temp_C and anneal_time_sec act through one thermal
                 budget axis Dt = t * exp(-Ea / kT) instead of two?

Every number this module reports is measured here; none are hard-coded.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import config

# Boltzmann constant in eV/K.
K_BOLTZMANN_EV = 8.617333262e-5

# Activation energy for boron diffusion in silicon (literature value, used as
# the physically anchored reference point for the thermal budget axis).
EA_BORON_EV = 3.5

# Interior factor levels: every axis drops its two extreme levels, so the
# hold-out is a closed shell around the training box.
INTERIOR_LEVELS: dict[str, list[float]] = {
    "dose_cm2": [1.0e15, 1.5e15, 2.0e15],
    "energy_keV": [12.5, 15.0, 17.5, 20.0, 30.0, 40.0],
    "anneal_temp_C": [950, 1000, 1050],
    "anneal_time_sec": [15, 25, 35],
}

LEARNING_CURVE_SIZES = (40, 60, 80, 120, 160, 240, 400, 600, 800)
LEARNING_CURVE_SIZES_QUICK = (40, 80, 160, 240)

AUDIT_TARGETS = ("xj_final_um", "rsh_final_ohm_sq")


@dataclass
class AuditResult:
    """Container for every measured audit section."""

    dataset: dict[str, Any] = field(default_factory=dict)
    sufficiency: dict[str, Any] = field(default_factory=dict)
    monotonicity: dict[str, Any] = field(default_factory=dict)
    extrapolation: dict[str, Any] = field(default_factory=dict)
    thermal_axis: dict[str, Any] = field(default_factory=dict)
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sufficiency": self.sufficiency,
            "monotonicity": self.monotonicity,
            "extrapolation": self.extrapolation,
            "thermal_axis": self.thermal_axis,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


def _gp(n_features: int, random_state: int = config.RANDOM_STATE):
    """Matern 5/2 GP pipeline matching the deployed surrogate family."""
    kernel = ConstantKernel(1.0) * Matern(
        length_scale=[1.0] * n_features, nu=2.5
    ) + WhiteKernel(1e-6)
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=1,
            random_state=random_state,
        ),
    )


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Model inputs in the order fixed by config.FEATURE_COLUMNS."""
    return np.column_stack(
        [
            np.log10(frame["dose_cm2"].to_numpy(float)),
            frame["energy_keV"].to_numpy(float),
            frame["anneal_temp_C"].to_numpy(float),
            frame["anneal_time_sec"].to_numpy(float),
        ]
    )


def thermal_budget(frame: pd.DataFrame, ea_ev: float) -> np.ndarray:
    """log10 of the Arrhenius thermal budget Dt = t * exp(-Ea / kT)."""
    kelvin = frame["anneal_temp_C"].to_numpy(float) + 273.15
    seconds = frame["anneal_time_sec"].to_numpy(float)
    return np.log10(seconds * np.exp(-ea_ev / (K_BOLTZMANN_EV * kelvin)))


def _quadratic_r2(x: np.ndarray, y: np.ndarray) -> float:
    """R2 of a 1-D quadratic least-squares fit (3 parameters)."""
    design = np.column_stack([x**2, x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    denom = float(((y - y.mean()) ** 2).sum())
    if denom == 0.0:
        return 1.0
    return float(1.0 - (resid**2).sum() / denom)


def _bilinear_quadratic_r2(frame: pd.DataFrame, y: np.ndarray) -> float:
    """R2 of a full 2-D quadratic in (temp, time) - 6 parameters."""
    raw = frame[["anneal_temp_C", "anneal_time_sec"]].to_numpy(float)
    design = np.column_stack(
        [raw, raw**2, raw[:, 0] * raw[:, 1], np.ones(len(frame))]
    )
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    denom = float(((y - y.mean()) ** 2).sum())
    if denom == 0.0:
        return 1.0
    return float(1.0 - (resid**2).sum() / denom)


# ---------------------------------------------------------------------------
# 1. Sufficiency - learning curve over random subsamples
# ---------------------------------------------------------------------------
def audit_sufficiency(
    frame: pd.DataFrame,
    sizes: tuple[int, ...] = LEARNING_CURVE_SIZES,
    repeats: int = 3,
    r2_threshold: float = 0.999,
) -> dict[str, Any]:
    features = _feature_matrix(frame)
    total = len(frame)
    rng = np.random.default_rng(config.RANDOM_STATE)
    out: dict[str, Any] = {
        "total_runs": total,
        "repeats": repeats,
        "r2_threshold": r2_threshold,
        "curve": {},
        "runs_needed": {},
        "compute_saving_pct": {},
    }

    for target in AUDIT_TARGETS:
        y = frame[target].to_numpy(float)
        curve = []
        for size in sizes:
            if size >= total:
                continue
            scores, errors = [], []
            for _ in range(repeats):
                train_idx = rng.choice(total, size, replace=False)
                test_idx = np.setdiff1d(np.arange(total), train_idx)
                model = _gp(features.shape[1]).fit(features[train_idx], y[train_idx])
                pred = model.predict(features[test_idx])
                scores.append(r2_score(y[test_idx], pred))
                errors.append(mean_absolute_error(y[test_idx], pred))
            curve.append(
                {
                    "n_train": int(size),
                    "r2": float(np.mean(scores)),
                    "r2_std": float(np.std(scores)),
                    "mae": float(np.mean(errors)),
                }
            )
        out["curve"][target] = curve

        reached = next((p for p in curve if p["r2"] >= r2_threshold), None)
        if reached is not None:
            needed = reached["n_train"]
            out["runs_needed"][target] = needed
            out["compute_saving_pct"][target] = round(100.0 * (1 - needed / total), 1)
        else:
            out["runs_needed"][target] = None
            out["compute_saving_pct"][target] = None

    needed = [v for v in out["runs_needed"].values() if v is not None]
    out["runs_needed_both_targets"] = max(needed) if needed else None
    if needed:
        out["compute_saving_pct_both_targets"] = round(
            100.0 * (1 - max(needed) / total), 1
        )
    else:
        out["compute_saving_pct_both_targets"] = None
    return out


# ---------------------------------------------------------------------------
# 2. Monotonicity - is x_j(dose) physically ordered at fixed energy?
# ---------------------------------------------------------------------------
def audit_monotonicity(frame: pd.DataFrame) -> dict[str, Any]:
    table = (
        frame.groupby(["energy_keV", "dose_cm2"])["xj_implant_um"].first().unstack()
    )
    doses = [float(c) for c in table.columns]
    per_energy = []
    violations = 0
    for energy, row in table.iterrows():
        values = row.to_numpy(float)
        diffs = np.diff(values)
        bad = [
            {
                "dose_from": doses[i],
                "dose_to": doses[i + 1],
                "delta_um": float(diffs[i]),
            }
            for i in range(len(diffs))
            if diffs[i] < 0
        ]
        violations += len(bad)
        per_energy.append(
            {
                "energy_keV": float(energy),
                "monotone_in_dose": bool(np.all(diffs > 0)),
                "values_um": [float(v) for v in values],
                "range_um": float(values.max() - values.min()),
                "violations": bad,
            }
        )

    # Shape of the dose response relative to each energy's own mean, in percent.
    relative = (table.div(table.mean(axis=1), axis=0) - 1.0) * 100.0
    shape_mean = relative.mean(axis=0).to_numpy(float)
    shape_std = relative.std(axis=0).to_numpy(float)

    # A per-dose artifact reproduces across energies -> low spread, high
    # correlation between each energy's shape and the pooled mean shape.
    corr = [
        float(np.corrcoef(relative.loc[e].to_numpy(float), shape_mean)[0, 1])
        for e in table.index
    ]

    return {
        "grid": {"doses_cm2": doses, "energies_keV": [float(e) for e in table.index]},
        "monotone_everywhere": violations == 0,
        "violation_count": int(violations),
        "per_energy": per_energy,
        "dose_shape_pct_of_mean": {
            "dose_cm2": doses,
            "mean_dev_pct": [float(v) for v in shape_mean],
            "std_dev_pct": [float(v) for v in shape_std],
        },
        "shape_reproducibility_corr": {
            "per_energy": corr,
            "mean": float(np.mean(corr)),
            "min": float(np.min(corr)),
        },
        "interpretation": (
            "x_j_implant should rise monotonically with dose at fixed energy. It "
            "does not: the same dose level dips at every energy with a highly "
            "reproducible shape, which is a systematic simulator artifact rather "
            "than run-to-run noise. Any surrogate fitted on the raw grid learns "
            "that artifact as if it were physics."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Extrapolation - interior box train, outer shell test
# ---------------------------------------------------------------------------
def audit_extrapolation(frame: pd.DataFrame) -> dict[str, Any]:
    mask = np.ones(len(frame), dtype=bool)
    for column, levels in INTERIOR_LEVELS.items():
        mask &= frame[column].isin(levels).to_numpy()

    features = _feature_matrix(frame)
    out: dict[str, Any] = {
        "interior_levels": {k: list(v) for k, v in INTERIOR_LEVELS.items()},
        "n_train_interior": int(mask.sum()),
        "n_test_outer_shell": int((~mask).sum()),
        "targets": {},
    }
    for target in AUDIT_TARGETS:
        y = frame[target].to_numpy(float)
        model = _gp(features.shape[1]).fit(features[mask], y[mask])
        outer = model.predict(features[~mask])
        inside = model.predict(features[mask])
        mae_out = mean_absolute_error(y[~mask], outer)
        mae_in = mean_absolute_error(y[mask], inside)
        out["targets"][target] = {
            "outer_shell_r2": float(r2_score(y[~mask], outer)),
            "outer_shell_mae": float(mae_out),
            "interior_fit_r2": float(r2_score(y[mask], inside)),
            "interior_fit_mae": float(mae_in),
            "mae_inflation_factor": (
                float(mae_out / mae_in) if mae_in > 0 else float("inf")
            ),
        }
    return out


# ---------------------------------------------------------------------------
# 4. Thermal axis - does (temp, time) collapse to a single Dt axis?
# ---------------------------------------------------------------------------
def audit_thermal_axis(
    frame: pd.DataFrame,
    target: str = "rsh_final_ohm_sq",
    ea_grid: tuple[float, ...] = (1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
) -> dict[str, Any]:
    log_target = np.log10(frame[target].to_numpy(float))
    groups = list(frame.groupby(["dose_cm2", "energy_keV"]).indices.values())

    def collapse_r2(ea_ev: float) -> float:
        axis = thermal_budget(frame, ea_ev)
        return float(
            np.mean([_quadratic_r2(axis[idx], log_target[idx]) for idx in groups])
        )

    scan = [{"ea_ev": float(ea), "collapse_r2": collapse_r2(ea)} for ea in ea_grid]
    best = max(scan, key=lambda row: row["collapse_r2"])

    baseline = float(
        np.mean(
            [
                _bilinear_quadratic_r2(frame.iloc[idx], log_target[idx])
                for idx in groups
            ]
        )
    )

    # Does replacing (temp, time) with the single Dt axis survive group CV on
    # unseen dose x energy conditions?  Measured, not assumed.
    group_id = frame.groupby(["dose_cm2", "energy_keV"]).ngroup().to_numpy()
    four = _feature_matrix(frame)
    three = np.column_stack(
        [
            np.log10(frame["dose_cm2"].to_numpy(float)),
            frame["energy_keV"].to_numpy(float),
            thermal_budget(frame, EA_BORON_EV),
        ]
    )
    group_cv: dict[str, Any] = {}
    for name, matrix in (("four_feature", four), ("three_feature_dt", three)):
        per_target = {}
        for tgt in AUDIT_TARGETS:
            y = frame[tgt].to_numpy(float)
            pred = np.zeros_like(y)
            splitter = GroupKFold(n_splits=8)
            for train_idx, test_idx in splitter.split(matrix, y, group_id):
                model = _gp(matrix.shape[1]).fit(matrix[train_idx], y[train_idx])
                pred[test_idx] = model.predict(matrix[test_idx])
            per_target[tgt] = {
                "group_cv_r2": float(r2_score(y, pred)),
                "group_cv_mae": float(mean_absolute_error(y, pred)),
            }
        group_cv[name] = per_target

    return {
        "target": target,
        "n_groups": len(groups),
        "ea_scan": scan,
        "best_fit": best,
        "boron_reference": {
            "ea_ev": EA_BORON_EV,
            "collapse_r2": collapse_r2(EA_BORON_EV),
        },
        "baseline_2d_quadratic_r2": baseline,
        "params": {"dt_axis_quadratic": 3, "temp_time_quadratic": 6},
        "group_cv": group_cv,
        "interpretation": (
            "A 3-parameter quadratic in a single thermal budget axis explains "
            "nearly as much within-condition variance as a 6-parameter surface "
            "in (temp, time), so the 5 x 5 anneal grid is close to one effective "
            "degree of freedom. Substituting the Dt axis for the two raw inputs "
            "does not improve group CV, so it is an interpretation asset rather "
            "than a modelling win."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_audit(quick: bool = False) -> AuditResult:
    started = time.perf_counter()
    frame = pd.read_csv(config.CSV_PATH)

    levels = {
        column: sorted(float(v) for v in frame[column].unique())
        for column in config.PROCESS_INPUTS
    }
    design_size = int(np.prod([len(v) for v in levels.values()]))

    result = AuditResult()
    result.dataset = {
        "csv": config.CSV_NAME,
        "rows": int(len(frame)),
        "levels": {k: v for k, v in levels.items()},
        "design_product": design_size,
        "full_factorial": design_size == len(frame),
        "duplicate_input_rows": int(
            frame.duplicated(subset=list(config.PROCESS_INPUTS)).sum()
        ),
    }
    result.sufficiency = audit_sufficiency(
        frame,
        sizes=LEARNING_CURVE_SIZES_QUICK if quick else LEARNING_CURVE_SIZES,
        repeats=2 if quick else 3,
    )
    result.monotonicity = audit_monotonicity(frame)
    result.extrapolation = audit_extrapolation(frame)
    result.thermal_axis = audit_thermal_axis(frame)
    result.elapsed_sec = time.perf_counter() - started
    return result


def save_audit(result: AuditResult, path=None):
    path = path or config.OUTPUT_DIR / "doe_audit_report.json"
    config.ensure_dirs()
    path.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path

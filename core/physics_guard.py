"""Physics-consistency guard for the surrogate.

Motivation
----------
The TCAD DOE is monotone in almost every direction.  Holding the other three
inputs fixed and stepping one axis at a time gives 11 (target, axis) pairs; ten
of them are perfectly ordered in the physically expected direction:

    x_j        rises with energy, rises with anneal temperature and time
    R_sh       falls with dose, energy, anneal temperature and time

The single exception is x_j versus dose, which violates the expected order in
about 28 % of its steps (225 of 800 for x_j_implant).  A deeper junction at
lower dose is not impossible in principle - pre-amorphisation at high dose
suppresses channelling and can pull x_j back - but that mechanism is monotone
in dose, so it would make the *highest* dose the shallowest.  What the data
shows instead is a single-level dip at 2.0e15 with 2.5e15 the deepest of all,
reproducing with correlation r = 0.94 across all eight energies.  A per-level
offset that repeats identically at every energy is a simulator artifact, not a
process effect, and an unconstrained surrogate happily learns it as physics.

What this module does
---------------------
It projects the trained surrogate onto the monotone cone along the dose axis
using pool-adjacent-violators (PAV), leaving every other axis untouched.  The
projection runs on a reference lattice once, is cached to disk, and is served
at query time by multilinear interpolation.

The raw surrogate is NOT replaced.  `core.surrogate.predict_raw` keeps its
original behaviour; this is an opt-in second opinion so the two can be compared
side by side, and `audit()` reports exactly what the constraint costs in
grid-reproduction error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from sklearn.isotonic import IsotonicRegression

from .config import MODEL_DIR, OUTPUT_DIR, ensure_dirs
from .dataset import input_bounds, load_dataset

GUARD_CACHE = MODEL_DIR / "physics_guard_lattice.npz"
GUARD_REPORT = OUTPUT_DIR / "physics_guard_report.json"

# Expected sign of d(target)/d(axis) implied by implant/anneal physics.
# +1 increasing, -1 decreasing, 0 no constraint asserted.
MONOTONE_EXPECTATION: dict[str, dict[str, int]] = {
    "xj_implant_um": {
        "dose_cm2": +1,
        "energy_keV": +1,
        "anneal_temp_C": 0,
        "anneal_time_sec": 0,
    },
    "xj_final_um": {
        "dose_cm2": +1,
        "energy_keV": +1,
        "anneal_temp_C": +1,
        "anneal_time_sec": +1,
    },
    "rsh_final_ohm_sq": {
        "dose_cm2": -1,
        "energy_keV": -1,
        "anneal_temp_C": -1,
        "anneal_time_sec": -1,
    },
}

# Only the dose axis is actually projected: it is the one axis the measured DOE
# disagrees with.  Constraining an axis the data already satisfies would be a
# no-op that silently costs accuracy.
GUARDED_AXIS = "dose_cm2"
GUARDED_TARGETS = ("xj_implant_um", "xj_final_um")

# Reference lattice resolution (dose, energy, temp, time).
LATTICE_SHAPE = (21, 17, 9, 9)

AXIS_ORDER = ("dose_cm2", "energy_keV", "anneal_temp_C", "anneal_time_sec")


def measure_monotonicity(frame=None) -> dict[str, Any]:
    """Count monotonicity violations per (target, axis) directly from the DOE."""
    df = load_dataset() if frame is None else frame
    out: dict[str, Any] = {}
    for target, expectations in MONOTONE_EXPECTATION.items():
        per_axis = {}
        for axis, expected in expectations.items():
            others = [a for a in AXIS_ORDER if a != axis]
            violations = steps = 0
            for _, group in df.groupby(others):
                ordered = group.sort_values(axis)[target].to_numpy(float)
                diffs = np.diff(ordered)
                steps += len(diffs)
                if expected > 0:
                    violations += int((diffs < 0).sum())
                elif expected < 0:
                    violations += int((diffs > 0).sum())
            per_axis[axis] = {
                "expected_sign": expected,
                "violations": violations,
                "steps": steps,
                "violation_rate": (violations / steps) if steps else 0.0,
                "asserted": expected != 0,
            }
        out[target] = per_axis
    return out


@dataclass
class GuardLattice:
    """Monotone-projected surrogate values on a regular reference lattice."""

    axes: tuple[np.ndarray, ...]
    values: dict[str, np.ndarray]
    raw_values: dict[str, np.ndarray]
    shape: tuple[int, ...]

    def interpolator(self, target: str) -> RegularGridInterpolator:
        return RegularGridInterpolator(
            self.axes,
            self.values[target],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )


def _lattice_axes(shape: tuple[int, ...] = LATTICE_SHAPE) -> tuple[np.ndarray, ...]:
    bounds = input_bounds()
    axes = []
    for axis, n in zip(AXIS_ORDER, shape):
        low = float(bounds[axis]["min"])
        high = float(bounds[axis]["max"])
        if axis == "dose_cm2":
            axes.append(np.logspace(np.log10(low), np.log10(high), n))
        else:
            axes.append(np.linspace(low, high, n))
    return tuple(axes)


def _isotonic_along_dose(block: np.ndarray, dose: np.ndarray, sign: int) -> np.ndarray:
    """PAV projection along axis 0 of a (n_dose, ...) block."""
    increasing = sign > 0
    flat = block.reshape(len(dose), -1)
    fixed = np.empty_like(flat)
    x = np.log10(dose)
    for column in range(flat.shape[1]):
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        fixed[:, column] = iso.fit_transform(x, flat[:, column])
    return fixed.reshape(block.shape)


def build_lattice(
    predict_fn: Callable[..., dict[str, np.ndarray]] | None = None,
    shape: tuple[int, ...] = LATTICE_SHAPE,
    targets: Iterable[str] = GUARDED_TARGETS,
) -> GuardLattice:
    """Evaluate the surrogate on the reference lattice and PAV-project it."""
    if predict_fn is None:
        from .surrogate import predict_raw as predict_fn  # local import: heavy

    axes = _lattice_axes(shape)
    mesh = np.meshgrid(*axes, indexing="ij")
    flat = [m.ravel() for m in mesh]
    predicted = predict_fn(flat[0], flat[1], flat[2], flat[3])

    raw_values: dict[str, np.ndarray] = {}
    values: dict[str, np.ndarray] = {}
    for target in targets:
        block = np.asarray(predicted[target], dtype=float).reshape(shape)
        raw_values[target] = block
        sign = MONOTONE_EXPECTATION[target][GUARDED_AXIS]
        values[target] = _isotonic_along_dose(block, axes[0], sign)
    return GuardLattice(axes=axes, values=values, raw_values=raw_values, shape=shape)


def save_lattice(lattice: GuardLattice, path=GUARD_CACHE) -> Any:
    ensure_dirs()
    payload = {f"axis_{i}": a for i, a in enumerate(lattice.axes)}
    for target, block in lattice.values.items():
        payload[f"guarded__{target}"] = block
    for target, block in lattice.raw_values.items():
        payload[f"raw__{target}"] = block
    np.savez_compressed(path, **payload)
    return path


def load_lattice(path=GUARD_CACHE) -> GuardLattice | None:
    if not path.exists():
        return None
    data = np.load(path)
    axes = tuple(data[f"axis_{i}"] for i in range(len(AXIS_ORDER)))
    values = {
        k.split("guarded__", 1)[1]: data[k]
        for k in data.files
        if k.startswith("guarded__")
    }
    raw_values = {
        k.split("raw__", 1)[1]: data[k] for k in data.files if k.startswith("raw__")
    }
    shape = tuple(len(a) for a in axes)
    return GuardLattice(axes=axes, values=values, raw_values=raw_values, shape=shape)


_GUARD: GuardLattice | None = None
_INTERP: dict[str, RegularGridInterpolator] = {}


def get_lattice(force: bool = False) -> GuardLattice:
    global _GUARD, _INTERP
    if force or _GUARD is None:
        _GUARD = load_lattice()
        if _GUARD is None or force:
            _GUARD = build_lattice()
            save_lattice(_GUARD)
        _INTERP = {}
    return _GUARD


def predict_guarded(
    dose: Any, energy: Any, temp: Any, time_s: Any
) -> dict[str, np.ndarray]:
    """Surrogate prediction with the dose-axis monotonicity constraint applied.

    Targets outside GUARDED_TARGETS are passed through from the raw surrogate
    unchanged, and delta_xj stays derived so the dataset identity is preserved.
    """
    from .surrogate import predict_raw

    lattice = get_lattice()
    raw = predict_raw(dose, energy, temp, time_s)
    query = np.column_stack(
        [
            np.atleast_1d(np.asarray(dose, dtype=float)),
            np.atleast_1d(np.asarray(energy, dtype=float)),
            np.atleast_1d(np.asarray(temp, dtype=float)),
            np.atleast_1d(np.asarray(time_s, dtype=float)),
        ]
    )
    out = dict(raw)
    for target in lattice.values:
        if target not in _INTERP:
            _INTERP[target] = lattice.interpolator(target)
        out[target] = _INTERP[target](query)
    out["delta_xj_um"] = out["xj_final_um"] - out["xj_implant_um"]
    return out


def audit(lattice: GuardLattice | None = None) -> dict[str, Any]:
    """What the constraint fixes, and what it costs, both measured."""
    from .surrogate import predict_raw

    lattice = lattice or get_lattice()
    df = load_dataset()

    doe = predict_raw(
        df["dose_cm2"].to_numpy(float),
        df["energy_keV"].to_numpy(float),
        df["anneal_temp_C"].to_numpy(float),
        df["anneal_time_sec"].to_numpy(float),
    )
    guarded = predict_guarded(
        df["dose_cm2"].to_numpy(float),
        df["energy_keV"].to_numpy(float),
        df["anneal_temp_C"].to_numpy(float),
        df["anneal_time_sec"].to_numpy(float),
    )

    def dose_violations(block: np.ndarray, sign: int) -> tuple[int, int]:
        diffs = np.diff(block, axis=0)
        bad = (diffs < 0).sum() if sign > 0 else (diffs > 0).sum()
        return int(bad), int(diffs.size)

    # The trained model's own honest error (GroupKFold on unseen dose x energy)
    # is the only fair yardstick for what the constraint costs: reproducing the
    # DOE grid exactly is memorisation of a deterministic simulation.
    group_cv_mae = _group_cv_mae()

    report: dict[str, Any] = {
        "dataset_monotonicity": measure_monotonicity(df),
        "lattice_shape": list(lattice.shape),
        "guarded_axis": GUARDED_AXIS,
        "targets": {},
    }
    for target in lattice.values:
        sign = MONOTONE_EXPECTATION[target][GUARDED_AXIS]
        before, total = dose_violations(lattice.raw_values[target], sign)
        after, _ = dose_violations(lattice.values[target], sign)
        truth = df[target].to_numpy(float)
        mae_raw = float(np.mean(np.abs(doe[target] - truth)))
        mae_guard = float(np.mean(np.abs(guarded[target] - truth)))
        cost = mae_guard - mae_raw
        max_shift = float(
            np.max(np.abs(lattice.values[target] - lattice.raw_values[target]))
        )
        span = float(truth.max() - truth.min())
        reference = group_cv_mae.get(target)
        report["targets"][target] = {
            "expected_sign": sign,
            "lattice_violations_before": before,
            "lattice_violations_after": after,
            "lattice_steps": total,
            "violation_rate_before": before / total if total else 0.0,
            "doe_mae_raw": mae_raw,
            "doe_mae_guarded": mae_guard,
            "doe_mae_cost": cost,
            "cost_pct_of_target_span": 100.0 * cost / span if span else None,
            "group_cv_mae": reference,
            "cost_vs_group_cv_mae": (cost / reference) if reference else None,
            "max_prediction_shift": max_shift,
        }
    return report


def _group_cv_mae() -> dict[str, float]:
    """Per-target GroupKFold MAE recorded by train_model.py, if available."""
    from .config import REPORT_PATH

    if not REPORT_PATH.exists():
        return {}
    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for target, block in payload.get("targets", {}).items():
        metrics = block.get("metrics", {}).get("group_cv", {})
        mae = metrics.get("mae")
        if mae is not None:
            out[target] = float(mae)
    return out


def save_report(report: dict[str, Any], path=GUARD_REPORT) -> Any:
    ensure_dirs()
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

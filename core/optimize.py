"""AI Process Optimizer.

MODE A - "Validated DOE Search"
    Ranks the 1,000 conditions that TCAD actually simulated, using the MEASURED
    (simulated) values from the CSV. Nothing is predicted, so every recommended
    recipe is backed by a real TCAD run.

MODE B - "AI Interpolation Search"
    Uses the trained surrogate to score a dense grid inside the DOE envelope
    (staged coarse search + local refinement), so recipes between simulated
    grid points can be proposed. These are AI interpolation results and are
    flagged as requiring TCAD / Fab verification.

Objective (multi-objective, user-weighted):
    score = w_xj * norm(|Xj_final - Xj_target|) + w_rsh * norm(Rsh)
Both normalisations use the observed TCAD data ranges, so the weights are
directly comparable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import input_bounds, load_dataset, target_bounds
from .surrogate import predict_raw

PARAM_ORDER = ["dose_cm2", "energy_keV", "anneal_temp_C", "anneal_time_sec"]


def _norm_ranges() -> tuple[float, float, float]:
    tb = target_bounds()
    xj_span = tb["xj_final_um"]["max"] - tb["xj_final_um"]["min"]
    rsh_lo = tb["rsh_final_ohm_sq"]["min"]
    rsh_hi = tb["rsh_final_ohm_sq"]["max"]
    return xj_span, rsh_lo, rsh_hi


def _score(
    xj: np.ndarray,
    rsh: np.ndarray,
    target_xj: float,
    w_xj: float,
    w_rsh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xj_span, rsh_lo, rsh_hi = _norm_ranges()
    err = np.abs(xj - target_xj)
    norm_err = err / (xj_span if xj_span > 0 else 1.0)
    norm_rsh = (rsh - rsh_lo) / ((rsh_hi - rsh_lo) or 1.0)
    total = (w_xj + w_rsh) or 1.0
    score = (w_xj * norm_err + w_rsh * np.clip(norm_rsh, 0.0, None)) / total
    return err, norm_err, norm_rsh, score


def _pareto_front(err: np.ndarray, rsh: np.ndarray, max_points: int = 400) -> np.ndarray:
    """Indices of the non-dominated set minimising (Xj error, Rsh)."""
    order = np.lexsort((rsh, err))
    front: list[int] = []
    best_rsh = np.inf
    for i in order:
        if rsh[i] < best_rsh - 1e-12:
            front.append(int(i))
            best_rsh = rsh[i]
    idx = np.array(front, dtype=int)
    if len(idx) > max_points:
        sel = np.linspace(0, len(idx) - 1, max_points).astype(int)
        idx = idx[sel]
    return idx


def _grid_axes(locks: dict[str, float] | None, resolution: dict[str, int]) -> dict[str, np.ndarray]:
    bounds = input_bounds()
    locks = locks or {}
    axes: dict[str, np.ndarray] = {}
    for name in PARAM_ORDER:
        if name in locks and locks[name] is not None:
            axes[name] = np.array([float(locks[name])], dtype=float)
            continue
        lo, hi = bounds[name]["min"], bounds[name]["max"]
        n = max(int(resolution.get(name, 13)), 1)
        if name == "dose_cm2":
            axes[name] = np.logspace(np.log10(lo), np.log10(hi), n)
        else:
            axes[name] = np.linspace(lo, hi, n)
    return axes


def _mesh(axes: dict[str, np.ndarray]) -> np.ndarray:
    grids = np.meshgrid(*[axes[p] for p in PARAM_ORDER], indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def _round_recipe(row: np.ndarray) -> dict[str, float]:
    return {
        "dose_cm2": float(f"{row[0]:.4g}"),
        "energy_keV": round(float(row[1]), 2),
        "anneal_temp_C": round(float(row[2]), 1),
        "anneal_time_sec": round(float(row[3]), 1),
    }


def _pack(
    params: np.ndarray,
    preds: dict[str, np.ndarray],
    err: np.ndarray,
    score: np.ndarray,
    feasible: np.ndarray,
    indices: np.ndarray,
    source: str,
    measured: dict[str, np.ndarray] | None = None,
    run_ids: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    items = []
    for rank, i in enumerate(indices, start=1):
        entry: dict[str, Any] = {
            "rank": rank,
            "source": source,
            "recipe": _round_recipe(params[i]),
            "predicted": {
                "xj_implant_um": float(preds["xj_implant_um"][i]),
                "xj_final_um": float(preds["xj_final_um"][i]),
                "delta_xj_um": float(preds["delta_xj_um"][i]),
                "rsh_final_ohm_sq": float(preds["rsh_final_ohm_sq"][i]),
            },
            "xj_error_um": float(err[i]),
            "score": float(score[i]),
            "feasible": bool(feasible[i]),
        }
        if measured is not None:
            entry["measured"] = {k: float(v[i]) for k, v in measured.items()}
        if run_ids is not None:
            entry["run_id"] = int(run_ids[i])
        items.append(entry)
    return items


def optimize(
    target_xj_um: float,
    tolerance_um: float = 0.01,
    rsh_mode: str = "minimize",
    rsh_max: float | None = None,
    w_xj: float = 0.6,
    w_rsh: float = 0.4,
    mode: str = "doe",
    top_k: int = 10,
    locks: dict[str, float] | None = None,
    refine: bool = True,
) -> dict[str, Any]:
    mode = (mode or "doe").lower()
    if mode not in {"doe", "ai"}:
        raise ValueError("mode must be 'doe' (Validated DOE Search) or 'ai' (AI Interpolation Search)")

    if mode == "doe":
        df = load_dataset()
        if locks:
            mask = np.ones(len(df), dtype=bool)
            for name, value in locks.items():
                if value is None:
                    continue
                col = df[name].to_numpy(dtype=float)
                mask &= np.isclose(col, float(value), rtol=1e-6, atol=1e-9)
            if mask.sum() == 0:
                mask[:] = True  # lock had no matching DOE level -> ignore it
            df = df[mask]
        params = df[PARAM_ORDER].to_numpy(dtype=float)
        measured = {
            "xj_implant_um": df["xj_implant_um"].to_numpy(dtype=float),
            "xj_final_um": df["xj_final_um"].to_numpy(dtype=float),
            "delta_xj_um": df["delta_xj_um"].to_numpy(dtype=float),
            "rsh_final_ohm_sq": df["rsh_final_ohm_sq"].to_numpy(dtype=float),
        }
        xj = measured["xj_final_um"]
        rsh = measured["rsh_final_ohm_sq"]
        surrogate = predict_raw(params[:, 0], params[:, 1], params[:, 2], params[:, 3])
        run_ids = df["run_id"].to_numpy(dtype=int)
        n_evaluated = int(len(df))
    else:
        # Coarse grid + local refinement: far fewer surrogate calls than a
        # single dense grid at the same effective resolution, which matters
        # because a kernel model (Gaussian Process) costs O(n_query x n_train).
        axes = _grid_axes(
            locks,
            {"dose_cm2": 11, "energy_keV": 13, "anneal_temp_C": 13, "anneal_time_sec": 13},
        )
        params = _mesh(axes)
        preds = predict_raw(params[:, 0], params[:, 1], params[:, 2], params[:, 3])
        n_evaluated = int(len(params))
        if refine:
            err0, _, _, score0 = _score(
                preds["xj_final_um"], preds["rsh_final_ohm_sq"], target_xj_um, w_xj, w_rsh
            )
            seeds = np.argsort(score0)[: min(20, len(score0))]
            steps = {
                name: (float(a[1] - a[0]) if len(a) > 1 else 0.0) for name, a in axes.items()
            }
            bounds = input_bounds()
            extra: list[np.ndarray] = []
            for s in seeds:
                loc_axes: dict[str, np.ndarray] = {}
                for k, name in enumerate(PARAM_ORDER):
                    if len(axes[name]) == 1:
                        loc_axes[name] = axes[name]
                        continue
                    if name == "dose_cm2":
                        c = np.log10(params[s, k])
                        step = float(np.log10(axes[name][1]) - np.log10(axes[name][0]))
                        g = np.clip(
                            np.linspace(c - step, c + step, 5),
                            np.log10(bounds[name]["min"]),
                            np.log10(bounds[name]["max"]),
                        )
                        loc_axes[name] = np.power(10.0, np.unique(g))
                    else:
                        c = params[s, k]
                        g = np.clip(
                            np.linspace(c - steps[name], c + steps[name], 5),
                            bounds[name]["min"],
                            bounds[name]["max"],
                        )
                        loc_axes[name] = np.unique(g)
                extra.append(_mesh(loc_axes))
            if extra:
                extra_params = np.unique(np.vstack(extra), axis=0)
                extra_preds = predict_raw(
                    extra_params[:, 0], extra_params[:, 1], extra_params[:, 2], extra_params[:, 3]
                )
                params = np.vstack([params, extra_params])
                preds = {k: np.concatenate([preds[k], extra_preds[k]]) for k in preds}
                n_evaluated += int(len(extra_params))
        surrogate = preds
        measured = None
        run_ids = None
        xj = preds["xj_final_um"]
        rsh = preds["rsh_final_ohm_sq"]

    err, norm_err, norm_rsh, score = _score(xj, rsh, target_xj_um, w_xj, w_rsh)

    feasible = err <= float(tolerance_um)
    constraint_note = None
    if rsh_mode == "constraint" and rsh_max is not None:
        feasible = feasible & (rsh <= float(rsh_max))
        constraint_note = f"Rsh <= {float(rsh_max):.4g} ohm/sq"

    order = np.lexsort((score, ~feasible))  # feasible first, then best score

    # The refinement stage can land on the same (or a near-identical) recipe
    # from several seeds. Recommendations are de-duplicated on the rounded
    # recipe, and in AI mode a minimum separation in normalised DOE space is
    # enforced so the Top-N are genuinely distinct process windows rather than
    # six neighbours of one point.
    bounds_n = input_bounds()
    def _norm_point(row: np.ndarray) -> np.ndarray:
        vals = []
        for k, name in enumerate(PARAM_ORDER):
            lo, hi = bounds_n[name]["min"], bounds_n[name]["max"]
            v = row[k]
            if name == "dose_cm2":
                lo, hi, v = np.log10(lo), np.log10(hi), np.log10(v)
            vals.append((v - lo) / ((hi - lo) or 1.0))
        return np.array(vals)

    min_sep = 0.06 if mode == "ai" else 0.0
    seen: set[tuple[float, ...]] = set()
    picked: list[int] = []
    picked_pts: list[np.ndarray] = []
    for i in order:
        key = tuple(_round_recipe(params[i]).values())
        if key in seen:
            continue
        if min_sep > 0:
            pt = _norm_point(params[i])
            if any(np.max(np.abs(pt - q)) < min_sep for q in picked_pts):
                continue
            picked_pts.append(pt)
        seen.add(key)
        picked.append(int(i))
        if len(picked) >= int(top_k):
            break
    top_idx = np.array(picked, dtype=int)

    pool = np.where(feasible)[0] if feasible.any() else np.arange(len(err))
    pf_local = _pareto_front(err[pool], rsh[pool])
    pareto_idx = pool[pf_local]

    recipes = _pack(
        params,
        surrogate if mode == "ai" else measured,  # DOE mode reports MEASURED values
        err,
        score,
        feasible,
        top_idx,
        source="TCAD measured (validated DOE point)" if mode == "doe" else "AI surrogate interpolation",
        measured=None,
        run_ids=run_ids,
    )
    if mode == "doe":
        for entry, i in zip(recipes, top_idx):
            entry["surrogate_check"] = {
                "xj_final_um": float(surrogate["xj_final_um"][i]),
                "rsh_final_ohm_sq": float(surrogate["rsh_final_ohm_sq"][i]),
                "xj_abs_error_um": float(abs(surrogate["xj_final_um"][i] - xj[i])),
                "rsh_abs_error_ohm_sq": float(abs(surrogate["rsh_final_ohm_sq"][i] - rsh[i])),
            }

    pareto = [
        {
            "recipe": _round_recipe(params[i]),
            "xj_final_um": float(xj[i]),
            "xj_error_um": float(err[i]),
            "rsh_final_ohm_sq": float(rsh[i]),
            "score": float(score[i]),
            "feasible": bool(feasible[i]),
        }
        for i in pareto_idx
    ]
    pareto.sort(key=lambda d: d["xj_error_um"])

    cloud_idx = np.argsort(score)[: min(1200, len(score))]
    cloud = {
        "xj_error_um": [float(v) for v in err[cloud_idx]],
        "rsh_final_ohm_sq": [float(v) for v in rsh[cloud_idx]],
    }

    return {
        "mode": mode,
        "mode_label": "MODE A · Validated DOE Search" if mode == "doe" else "MODE B · AI Interpolation Search",
        "verification_warning": None
        if mode == "doe"
        else "AI interpolation result — TCAD/Fab verification required",
        "verification_warning_ko": None
        if mode == "doe"
        else "AI 보간 결과입니다 — 반드시 TCAD 재시뮬레이션 또는 Fab 검증이 필요합니다.",
        "objective": {
            "target_xj_um": float(target_xj_um),
            "tolerance_um": float(tolerance_um),
            "rsh_mode": rsh_mode,
            "rsh_max": float(rsh_max) if rsh_max is not None else None,
            "w_xj": float(w_xj),
            "w_rsh": float(w_rsh),
            "constraint_note": constraint_note,
            "formula": "score = (w_xj * |Xj-Xj_target|/range(Xj) + w_rsh * (Rsh-min)/range(Rsh)) / (w_xj+w_rsh)",
        },
        "search": {
            "candidates_evaluated": n_evaluated,
            "feasible_count": int(feasible.sum()),
            "locks": {k: v for k, v in (locks or {}).items() if v is not None},
        },
        "recipes": recipes,
        "pareto": pareto,
        "cloud": cloud,
        "achievable_range": {
            "xj_final_um": [float(np.min(xj)), float(np.max(xj))],
            "rsh_final_ohm_sq": [float(np.min(rsh)), float(np.max(rsh))],
        },
    }

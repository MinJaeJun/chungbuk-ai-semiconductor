"""Robustness / process-window analysis under real equipment scatter.

The TCAD DOE is deterministic, so a nominal optimum is a single-point answer.
A real line applies scatter to every setpoint, and the question a process
engineer actually asks is:

    "If I load this recipe, what fraction of wafers still meets spec?"

This module answers that by Monte-Carlo sampling the four process inputs around
a nominal recipe, pushing every sample through the trained surrogate, and
reporting spec yield, Cpk and a variance decomposition that says WHICH input's
scatter dominates the output scatter.

Tolerance presets are anchored to variation actually measured on a real
production line (core/variation.py). That reference process is PECVD SiON, not
ion implantation, so the presets are order-of-magnitude anchors only and every
tolerance is user-overridable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import input_bounds
from .surrogate import predict_raw

PARAMS = ["dose_cm2", "energy_keV", "anneal_temp_C", "anneal_time_sec"]

# Relative 1-sigma (%) per process input.
#   - "measured_w2w"  : mean wafer-to-wafer CV observed in-line (0.65 %)
#   - "measured_p90"  : 90th percentile of that same distribution (1.38 %)
#   - "conservative"  : pooled total CV across every wafer and recipe (2.04 %)
# Anneal temperature is expressed relative to setpoint; 0.5 % of 1000 degC = 5 degC,
# which is a realistic RTA control band and is kept below the deposition-derived
# figure on purpose (RTA pyrometer control is tighter than film thickness).
TOLERANCE_PRESETS: dict[str, dict[str, Any]] = {
    "tight": {
        "label": "Tight · 장비 제어 양호",
        "anchor": "measured wafer-to-wafer CV (mean)",
        "cv_pct": {"dose_cm2": 0.65, "energy_keV": 0.30, "anneal_temp_C": 0.25, "anneal_time_sec": 1.0},
    },
    "typical": {
        "label": "Typical · 실측 p90 기준",
        "anchor": "measured wafer-to-wafer CV (p90)",
        "cv_pct": {"dose_cm2": 1.40, "energy_keV": 0.60, "anneal_temp_C": 0.50, "anneal_time_sec": 2.0},
    },
    "conservative": {
        "label": "Conservative · 실측 total CV 기준",
        "anchor": "measured pooled total CV across all wafers/recipes",
        "cv_pct": {"dose_cm2": 2.04, "energy_keV": 1.00, "anneal_temp_C": 0.80, "anneal_time_sec": 4.0},
    },
}
DEFAULT_PRESET = "typical"


def resolve_tolerances(
    preset: str | None = None, cv_pct: dict[str, float] | None = None
) -> dict[str, Any]:
    base = TOLERANCE_PRESETS.get(preset or DEFAULT_PRESET, TOLERANCE_PRESETS[DEFAULT_PRESET])
    resolved = dict(base["cv_pct"])
    custom = False
    if cv_pct:
        for k, v in cv_pct.items():
            if k in resolved and v is not None:
                # The UI echoes the preset values back on every call, so only a
                # real deviation counts as a custom tolerance.
                if abs(float(v) - resolved[k]) > 1e-9:
                    custom = True
                resolved[k] = float(v)
    return {
        "preset": preset or DEFAULT_PRESET,
        "label": ("Custom · 사용자 지정" if custom else base["label"]),
        "anchor": base["anchor"],
        "cv_pct": resolved,
        "is_custom": custom,
    }


def _sample(
    nominal: dict[str, float],
    cv_pct: dict[str, float],
    n: int,
    rng: np.random.Generator,
    only: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Gaussian scatter around the setpoint, clipped to the training envelope.

    Clipping matters: a sample outside the DOE box would be an extrapolation
    the surrogate cannot honour, so the analysis would silently report
    fabricated numbers.

    But clipping is not free. A setpoint sitting on a DOE corner (dose at
    2.5e15, say) has half of its scatter truncated, which shrinks the apparent
    output variance and inflates the yield estimate. Hiding that would be a
    silent lie, so the clipped fraction is measured and returned for reporting.
    """
    bounds = input_bounds()
    out: dict[str, np.ndarray] = {}
    clipped: dict[str, float] = {}
    for p in PARAMS:
        mu = float(nominal[p])
        if only is not None and p != only:
            out[p] = np.full(n, mu)
            clipped[p] = 0.0
            continue
        sigma = abs(mu) * (float(cv_pct[p]) / 100.0)
        draw = rng.normal(mu, sigma, n) if sigma > 0 else np.full(n, mu)
        lo, hi = bounds[p]["min"], bounds[p]["max"]
        clipped[p] = float(np.count_nonzero((draw < lo) | (draw > hi)) / n * 100.0)
        out[p] = np.clip(draw, lo, hi)
    return out, clipped


def _cpk_two_sided(mean: float, sigma: float, lsl: float, usl: float) -> float | None:
    if sigma <= 0:
        return None
    return float(min(usl - mean, mean - lsl) / (3.0 * sigma))


def _cpk_upper(mean: float, sigma: float, usl: float) -> float | None:
    if sigma <= 0:
        return None
    return float((usl - mean) / (3.0 * sigma))


def _hist(values: np.ndarray, bins: int = 34) -> dict[str, list[float]]:
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {
        "centers": [float(v) for v in centers],
        "counts": [int(v) for v in counts],
        "edges": [float(v) for v in edges],
    }


def robust_analysis(
    dose_cm2: float,
    energy_keV: float,
    anneal_temp_C: float,
    anneal_time_sec: float,
    target_xj_um: float,
    tolerance_um: float = 0.01,
    rsh_max: float | None = None,
    preset: str | None = None,
    cv_pct: dict[str, float] | None = None,
    n_samples: int = 2500,
    seed: int = 20260216,
) -> dict[str, Any]:
    """Monte-Carlo spec yield + Cpk + variance decomposition for one recipe."""
    nominal = {
        "dose_cm2": float(dose_cm2),
        "energy_keV": float(energy_keV),
        "anneal_temp_C": float(anneal_temp_C),
        "anneal_time_sec": float(anneal_time_sec),
    }
    tol = resolve_tolerances(preset, cv_pct)
    n = int(max(200, min(n_samples, 20000)))
    rng = np.random.default_rng(seed)
    # The decomposition needs 4 extra sweeps; half the budget is plenty for
    # a variance SHARE, and it keeps the endpoint interactive.
    n_dec = max(400, n // 2)

    s, clipped = _sample(nominal, tol["cv_pct"], n, rng)
    pred = predict_raw(s["dose_cm2"], s["energy_keV"], s["anneal_temp_C"], s["anneal_time_sec"])
    xj, rsh = pred["xj_final_um"], pred["rsh_final_ohm_sq"]

    nom = predict_raw(*(nominal[p] for p in PARAMS))
    nominal_pred = {k: float(v[0]) for k, v in nom.items()}

    lsl, usl = target_xj_um - tolerance_um, target_xj_um + tolerance_um
    in_xj = (xj >= lsl) & (xj <= usl)
    yield_xj = float(in_xj.mean() * 100)

    if rsh_max is not None:
        in_rsh = rsh <= float(rsh_max)
        yield_rsh = float(in_rsh.mean() * 100)
        joint = float((in_xj & in_rsh).mean() * 100)
        cpk_rsh = _cpk_upper(float(rsh.mean()), float(rsh.std(ddof=1)), float(rsh_max))
    else:
        yield_rsh, joint, cpk_rsh = None, yield_xj, None

    # ---- first-order variance decomposition ------------------------------
    # Vary one input at a time to see whose scatter drives the output scatter.
    contrib: dict[str, dict[str, float]] = {}
    for p in PARAMS:
        s1, _ = _sample(nominal, tol["cv_pct"], n_dec, np.random.default_rng(seed + 17), only=p)
        q = predict_raw(s1["dose_cm2"], s1["energy_keV"], s1["anneal_temp_C"], s1["anneal_time_sec"])
        contrib[p] = {
            "xj_var": float(np.var(q["xj_final_um"], ddof=1)),
            "rsh_var": float(np.var(q["rsh_final_ohm_sq"], ddof=1)),
            "xj_std": float(np.std(q["xj_final_um"], ddof=1)),
            "rsh_std": float(np.std(q["rsh_final_ohm_sq"], ddof=1)),
        }
    tot_xj = sum(c["xj_var"] for c in contrib.values()) or 1.0
    tot_rsh = sum(c["rsh_var"] for c in contrib.values()) or 1.0
    decomposition = sorted(
        (
            {
                "parameter": p,
                "xj_share_pct": contrib[p]["xj_var"] / tot_xj * 100,
                "rsh_share_pct": contrib[p]["rsh_var"] / tot_rsh * 100,
                "xj_std": contrib[p]["xj_std"],
                "rsh_std": contrib[p]["rsh_std"],
                "cv_pct": tol["cv_pct"][p],
            }
            for p in PARAMS
        ),
        key=lambda d: -d["xj_share_pct"],
    )

    def dist(v: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)),
            "cv_pct": float(v.std(ddof=1) / abs(v.mean()) * 100) if v.mean() != 0 else None,
            "min": float(v.min()),
            "max": float(v.max()),
            "p01": float(np.percentile(v, 1)),
            "p05": float(np.percentile(v, 5)),
            "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)),
            "histogram": _hist(v),
        }

    return {
        "nominal": nominal,
        "nominal_prediction": nominal_pred,
        "tolerances": tol,
        "n_samples": n,
        "boundary_clipping": {
            "pct_by_parameter": clipped,
            "max_pct": max(clipped.values()) if clipped else 0.0,
            "significant": bool(max(clipped.values()) > 1.0) if clipped else False,
            "note_ko": (
                "설정점이 DOE 경계에 있으면 산포 샘플이 학습 범위 밖으로 나가 clip됩니다. "
                "이 경우 실제 산포보다 좁게 평가되어 수율이 과대평가될 수 있습니다."
            ),
            "note_en": (
                "Setpoints on the DOE boundary have part of their scatter clipped, which "
                "under-states real variance and can over-state yield."
            ),
        },
        "spec": {
            "target_xj_um": float(target_xj_um),
            "tolerance_um": float(tolerance_um),
            "lsl": float(lsl),
            "usl": float(usl),
            "rsh_max": float(rsh_max) if rsh_max is not None else None,
        },
        "yield": {
            "xj_pct": yield_xj,
            "rsh_pct": yield_rsh,
            "joint_pct": joint,
        },
        "cpk": {
            "xj": _cpk_two_sided(float(xj.mean()), float(xj.std(ddof=1)), lsl, usl),
            "rsh": cpk_rsh,
        },
        "distribution": {"xj_final_um": dist(xj), "rsh_final_ohm_sq": dist(rsh)},
        "variance_decomposition": decomposition,
        "note_ko": (
            "공정 입력 산포를 가우시안으로 가정하고 학습 범위 내부로 clip하여 surrogate에 통과시킨 "
            "Monte-Carlo 결과입니다. 산포 크기는 실측 라인 기준값을 참조했으나 이온주입 공정의 "
            "실제 장비 산포는 아니며, 반드시 해당 장비의 실측 tolerance로 교체해야 합니다."
        ),
        "note_en": (
            "Monte-Carlo over Gaussian input scatter, clipped to the training envelope. "
            "Tolerance magnitudes are anchored to a real in-line metrology dataset from a "
            "different process and must be replaced with the actual implanter tolerances."
        ),
    }


def robust_rank(
    recipes: list[dict[str, float]],
    target_xj_um: float,
    tolerance_um: float = 0.01,
    rsh_max: float | None = None,
    preset: str | None = None,
    cv_pct: dict[str, float] | None = None,
    n_samples: int = 800,
    seed: int = 20260216,
) -> list[dict[str, Any]]:
    """Spec yield for each candidate recipe, ranked robust-first.

    The nominal optimum and the robust optimum are frequently NOT the same
    recipe: a condition sitting on a steep part of the response surface can
    hit the target exactly and still lose most of its wafers to scatter.
    """
    tol = resolve_tolerances(preset, cv_pct)
    n = int(max(200, min(n_samples, 8000)))
    lsl, usl = target_xj_um - tolerance_um, target_xj_um + tolerance_um

    out: list[dict[str, Any]] = []
    for i, rec in enumerate(recipes):
        nominal = {p: float(rec[p]) for p in PARAMS}
        s, _ = _sample(nominal, tol["cv_pct"], n, np.random.default_rng(seed + i))
        q = predict_raw(s["dose_cm2"], s["energy_keV"], s["anneal_temp_C"], s["anneal_time_sec"])
        xj, rsh = q["xj_final_um"], q["rsh_final_ohm_sq"]
        in_xj = (xj >= lsl) & (xj <= usl)
        in_rsh = rsh <= float(rsh_max) if rsh_max is not None else np.ones(n, dtype=bool)
        out.append(
            {
                "recipe": nominal,
                "nominal_rank": i + 1,
                "yield_xj_pct": float(in_xj.mean() * 100),
                "yield_joint_pct": float((in_xj & in_rsh).mean() * 100),
                "xj_mean": float(xj.mean()),
                "xj_std": float(xj.std(ddof=1)),
                "rsh_mean": float(rsh.mean()),
                "rsh_std": float(rsh.std(ddof=1)),
                "cpk_xj": _cpk_two_sided(float(xj.mean()), float(xj.std(ddof=1)), lsl, usl),
            }
        )
    out.sort(key=lambda d: (-d["yield_joint_pct"], d["xj_std"]))
    for rank, item in enumerate(out, start=1):
        item["robust_rank"] = rank
        item["rank_shift"] = item["nominal_rank"] - rank
    return out

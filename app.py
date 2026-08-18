"""FastAPI backend for the AI Semiconductor Process Optimizer.

Frontend (static/index.html)  --REST-->  FastAPI  -->  ML surrogate  -->  TCAD CSV

Run:
    uvicorn app:app --reload --port 8000
    python app.py                     (equivalent, no reload)
"""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import chungbuk, llm, physics_guard, surrogate
from core.config import (
    DISCLAIMER_EN,
    DISCLAIMER_KO,
    INPUT_META,
    MODEL_TARGETS,
    OUTPUT_DIR,
    PROCESS_INPUTS,
    STATIC_DIR,
    TARGET_META,
)
from core.dataset import (
    correlation_matrix,
    doe_levels,
    input_bounds,
    load_dataset,
    marginal_effects,
    scatter_payload,
    target_bounds,
    validate_dataset,
)
from core.explain import global_explanation, local_explanation
from core.insight import build_insights, data_observations, optimizer_insights
from core.optimize import optimize as run_optimize
from core.robust import (
    TOLERANCE_PRESETS,
    resolve_tolerances,
    robust_analysis,
    robust_rank,
)
from core.surrogate import (
    SurrogateNotTrained,
    delta_resolution,
    extrapolation_report,
    is_trained,
    load_bundle,
    model_uncertainty,
    nearest_doe_run,
    predict_raw,
)
from core import variation
from core.whatif import sweep as run_sweep

app = FastAPI(
    title="AI Semiconductor Process Optimizer",
    description=(
        "TCAD-based Ion Implantation & Annealing process prediction / optimization "
        "platform. AI surrogate for process decision support - not a replacement "
        "for TCAD or Fab qualification."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# request models
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    dose_cm2: float = Field(..., gt=0, description="Implant dose [cm^-2]")
    energy_keV: float = Field(..., gt=0, description="Implant energy [keV]")
    anneal_temp_C: float = Field(..., gt=0, description="Anneal temperature [degC]")
    anneal_time_sec: float = Field(..., gt=0, description="Anneal time [sec]")
    explain: bool = True


class SweepRequest(PredictRequest):
    parameter: Literal["dose_cm2", "energy_keV", "anneal_temp_C", "anneal_time_sec"] = "anneal_temp_C"
    n_points: int = Field(61, ge=5, le=401)
    lo: float | None = None
    hi: float | None = None


class OptimizeRequest(BaseModel):
    target_xj_um: float = Field(..., gt=0)
    tolerance_um: float = Field(0.01, gt=0)
    rsh_mode: Literal["minimize", "constraint"] = "minimize"
    rsh_max: float | None = None
    w_xj: float = Field(0.6, ge=0, le=1)
    w_rsh: float = Field(0.4, ge=0, le=1)
    mode: Literal["doe", "ai"] = "doe"
    top_k: int = Field(10, ge=1, le=50)
    lock_dose_cm2: float | None = None
    lock_energy_keV: float | None = None
    lock_anneal_temp_C: float | None = None
    lock_anneal_time_sec: float | None = None
    # Re-rank the Top-N by Monte-Carlo spec yield instead of nominal score.
    robust_rerank: bool = False
    robust_preset: Literal["tight", "typical", "conservative"] = "typical"


class RobustRequest(BaseModel):
    dose_cm2: float = Field(..., gt=0)
    energy_keV: float = Field(..., gt=0)
    anneal_temp_C: float = Field(..., gt=0)
    anneal_time_sec: float = Field(..., gt=0)
    target_xj_um: float = Field(..., gt=0)
    tolerance_um: float = Field(0.01, gt=0)
    rsh_max: float | None = None
    preset: Literal["tight", "typical", "conservative"] = "typical"
    cv_pct: dict[str, float] | None = None
    n_samples: int = Field(3000, ge=200, le=20000)


class NarrateRequest(BaseModel):
    evidence: dict[str, Any]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require_model() -> dict[str, Any]:
    try:
        return load_bundle()
    except SurrogateNotTrained as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _metrics_by_target() -> dict[str, Any]:
    report = _require_model()["report"]
    return {t: report["targets"][t]["metrics"] for t in MODEL_TARGETS}


def _delta_flag(delta: float) -> dict[str, Any]:
    """delta_xj is a small difference of two predictions - report its resolution."""
    res = delta_resolution()
    limit = res["resolution_limit"]
    resolved = abs(delta) >= limit
    return {
        **res,
        "resolved": bool(resolved),
        "message_ko": (
            "예측 ΔXj가 surrogate 분해능보다 크므로 확산 경향을 해석할 수 있습니다."
            if resolved
            else f"예측 ΔXj({delta:.3g} um)가 surrogate 분해능({limit:.3g} um)보다 작습니다. "
            "이 조건의 확산량은 모델 오차 수준이므로 값 자체를 해석하지 마십시오."
        ),
        "message_en": (
            "Derived delta_xj exceeds the surrogate resolution limit."
            if resolved
            else f"Derived delta_xj ({delta:.3g} um) is below the surrogate resolution "
            f"limit ({limit:.3g} um); treat it as unresolved."
        ),
    }


# --------------------------------------------------------------------------- #
# meta / dataset
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_trained": is_trained(),
        "llm": llm.available(),
    }


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    df = load_dataset()
    payload: dict[str, Any] = {
        "app": {
            "name_en": "AI Semiconductor Process Optimizer",
            "name_ko": "AI 기반 반도체 이온주입·열처리 공정 최적화 플랫폼",
            "subtitle_en": "TCAD-based Ion Implantation & Annealing Process Prediction / Optimization Platform",
            "tagline": "From Simulation Data to Optimized Recipe",
            "positioning_ko": (
                "TCAD 기반 공정 의사결정 지원용 AI Surrogate Model — "
                "공정 조건 후보를 빠르게 탐색하기 위한 AI 지원 시스템"
            ),
        },
        "dataset": {
            "rows": int(len(df)),
            "inputs": PROCESS_INPUTS,
            "input_meta": INPUT_META,
            "targets": MODEL_TARGETS + ["delta_xj_um"],
            "target_meta": TARGET_META,
            "doe_levels": doe_levels(df),
            "input_bounds": input_bounds(df),
            "target_bounds": target_bounds(df),
        },
        "model_trained": is_trained(),
        "disclaimer_en": DISCLAIMER_EN,
        "disclaimer_ko": DISCLAIMER_KO,
        "llm": llm.available(),
    }
    if is_trained():
        report = load_bundle()["report"]
        payload["model"] = {
            "trained_at": report["generated_at"],
            "environment": report["environment"],
            "best_models": {t: report["targets"][t]["best_model"] for t in MODEL_TARGETS},
            "metrics": _metrics_by_target(),
            "selection_metrics": {
                t: report["targets"][t]["selection_metric"] for t in MODEL_TARGETS
            },
            "derived_delta": report.get("derived_delta", {}).get("test", {}),
            "n_train": report["targets"][MODEL_TARGETS[0]]["n_train"],
            "n_test": report["targets"][MODEL_TARGETS[0]]["n_test"],
            "derived_targets": report["derived_targets"],
        }
    return payload


@app.get("/api/dataset/summary")
def dataset_summary() -> dict[str, Any]:
    df = load_dataset()
    return {
        "validation": validate_dataset(df),
        "correlation": correlation_matrix(df),
        "marginal_effects": marginal_effects(df),
        "observations": data_observations(),
    }


@app.get("/api/dataset/points")
def dataset_points() -> dict[str, Any]:
    return {"columns": scatter_payload()}


# --------------------------------------------------------------------------- #
# prediction / what-if
# --------------------------------------------------------------------------- #
@app.post("/api/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    _require_model()
    preds = predict_raw(req.dose_cm2, req.energy_keV, req.anneal_temp_C, req.anneal_time_sec)
    extrap = extrapolation_report(
        req.dose_cm2, req.energy_keV, req.anneal_temp_C, req.anneal_time_sec
    )
    nearest = nearest_doe_run(
        req.dose_cm2, req.energy_keV, req.anneal_temp_C, req.anneal_time_sec
    )
    metrics = _metrics_by_target()

    prediction = {
        "xj_implant_um": float(preds["xj_implant_um"][0]),
        "xj_final_um": float(preds["xj_final_um"][0]),
        "delta_xj_um": float(preds["delta_xj_um"][0]),
        "rsh_final_ohm_sq": float(preds["rsh_final_ohm_sq"][0]),
    }
    # Uncertainty band shown to the user = held-out test RMSE of that target.
    uncertainty = {
        t: {
            "test_rmse": metrics[t]["test"]["rmse"],
            "test_mae": metrics[t]["test"]["mae"],
            "group_cv_rmse": metrics[t]["group_cv"]["rmse"],
        }
        for t in MODEL_TARGETS
    }

    explanation: dict[str, Any] = {}
    insights: list[dict[str, Any]] = []
    if req.explain:
        explanation = local_explanation(
            req.dose_cm2, req.energy_keV, req.anneal_temp_C, req.anneal_time_sec
        )
        insights = build_insights(prediction, explanation, extrap, metrics, nearest)

    return {
        "input": {
            "dose_cm2": req.dose_cm2,
            "energy_keV": req.energy_keV,
            "anneal_temp_C": req.anneal_temp_C,
            "anneal_time_sec": req.anneal_time_sec,
        },
        "prediction": prediction,
        "units": {t: TARGET_META[t]["unit"] for t in MODEL_TARGETS + ["delta_xj_um"]},
        "uncertainty": uncertainty,
        "delta_resolution": _delta_flag(prediction["delta_xj_um"]),
        "extrapolation": extrap,
        "nearest_doe_run": nearest,
        "explanation": {k: v for k, v in explanation.items() if not k.startswith("_")},
        "insights": insights,
        "note": "delta_xj_um is derived as (predicted xj_final_um - predicted xj_implant_um).",
    }


@app.post("/api/whatif")
def whatif(req: SweepRequest) -> dict[str, Any]:
    _require_model()
    return run_sweep(
        req.parameter,
        req.dose_cm2,
        req.energy_keV,
        req.anneal_temp_C,
        req.anneal_time_sec,
        n_points=req.n_points,
        lo=req.lo,
        hi=req.hi,
    )


# --------------------------------------------------------------------------- #
# optimizer
# --------------------------------------------------------------------------- #
@app.post("/api/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    _require_model()
    locks = {
        "dose_cm2": req.lock_dose_cm2,
        "energy_keV": req.lock_energy_keV,
        "anneal_temp_C": req.lock_anneal_temp_C,
        "anneal_time_sec": req.lock_anneal_time_sec,
    }
    result = run_optimize(
        target_xj_um=req.target_xj_um,
        tolerance_um=req.tolerance_um,
        rsh_mode=req.rsh_mode,
        rsh_max=req.rsh_max,
        w_xj=req.w_xj,
        w_rsh=req.w_rsh,
        mode=req.mode,
        top_k=req.top_k,
        locks=locks,
    )
    if req.robust_rerank and result["recipes"]:
        result["robust"] = {
            "ranking": robust_rank(
                [r["recipe"] for r in result["recipes"]],
                target_xj_um=req.target_xj_um,
                tolerance_um=req.tolerance_um,
                rsh_max=req.rsh_max if req.rsh_mode == "constraint" else None,
                preset=req.robust_preset,
            ),
            "tolerances": resolve_tolerances(req.robust_preset),
        }
    result["insights"] = optimizer_insights(result)
    return result


# --------------------------------------------------------------------------- #
# robustness under real equipment scatter
# --------------------------------------------------------------------------- #
@app.get("/api/variation")
def variation_reference() -> dict[str, Any]:
    """Process variation measured on a real production line (reference only)."""
    if not variation.available():
        return {
            "available": False,
            "detail": "fab_thickness_profile_17000.csv not found in data/",
        }
    return {"available": True, "presets": TOLERANCE_PRESETS, **variation.analyze()}


@app.post("/api/robust")
def robust(req: RobustRequest) -> dict[str, Any]:
    _require_model()
    result = robust_analysis(
        dose_cm2=req.dose_cm2,
        energy_keV=req.energy_keV,
        anneal_temp_C=req.anneal_temp_C,
        anneal_time_sec=req.anneal_time_sec,
        target_xj_um=req.target_xj_um,
        tolerance_um=req.tolerance_um,
        rsh_max=req.rsh_max,
        preset=req.preset,
        cv_pct=req.cv_pct,
        n_samples=req.n_samples,
    )
    unc = model_uncertainty(
        req.dose_cm2, req.energy_keV, req.anneal_temp_C, req.anneal_time_sec
    )
    result["model_uncertainty"] = {
        t: {"std": v["std"][0], "source": v["source"], "test_rmse": v["test_rmse"]}
        for t, v in unc.items()
    }
    return result


# --------------------------------------------------------------------------- #
# XAI / validation
# --------------------------------------------------------------------------- #
@app.get("/api/xai/global")
def xai_global() -> dict[str, Any]:
    _require_model()
    return global_explanation()


@app.get("/api/validation")
def validation() -> dict[str, Any]:
    report = _require_model()["report"]
    out: dict[str, Any] = {
        "generated_at": report["generated_at"],
        "environment": report["environment"],
        "dataset_coverage": {
            "total_runs": report["dataset"]["n_rows"],
            "doe": report["dataset"]["doe"],
            "input_bounds": report["dataset"]["input_bounds"],
            "target_bounds": report["dataset"]["target_bounds"],
            "leakage_policy": report["dataset"]["leakage_policy"],
        },
        "disclaimer_en": report["disclaimer_en"],
        "disclaimer_ko": report["disclaimer_ko"],
        "targets": {},
    }
    for t in MODEL_TARGETS:
        block = report["targets"][t]
        out["targets"][t] = {
            "label": TARGET_META[t]["label"],
            "label_ko": TARGET_META[t]["label_ko"],
            "unit": TARGET_META[t]["unit"],
            "best_model": block["best_model"],
            "selection_metric": block["selection_metric"],
            "n_train": block["n_train"],
            "n_test": block["n_test"],
            "n_total": block["n_total"],
            "n_groups": block["n_groups"],
            "cv_folds": block["cv_folds"],
            "metrics": block["metrics"],
            "leaderboard": block["leaderboard"],
            "validation": block["validation"],
            "group_validation": block["group_validation"],
        }
    return out

# --------------------------------------------------------------------------- #
# DOE audit / physics guard / regional fit
# --------------------------------------------------------------------------- #
@app.get("/api/doe/audit")
def doe_audit_report() -> dict[str, Any]:
    """Cached DOE information-content audit (run `python audit_doe.py` to build)."""
    path = OUTPUT_DIR / "doe_audit_report.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="DOE audit not generated yet. Run: python audit_doe.py",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/physics/guard")
def physics_guard_report() -> dict[str, Any]:
    """Monotonicity audit plus what the dose-axis constraint fixes and costs."""
    _require_model()
    return physics_guard.audit()


@app.post("/api/physics/compare")
def physics_compare(req: PredictRequest) -> dict[str, Any]:
    """Raw surrogate vs dose-monotone guarded surrogate at one condition."""
    _require_model()
    args = (req.dose_cm2, req.energy_keV, req.anneal_temp_C, req.anneal_time_sec)
    raw = surrogate.predict_raw(*args)
    guarded = physics_guard.predict_guarded(*args)
    out: dict[str, Any] = {"inputs": req.model_dump(), "targets": {}}
    for target in ("xj_implant_um", "xj_final_um", "delta_xj_um"):
        raw_value = float(np.asarray(raw[target]).ravel()[0])
        guard_value = float(np.asarray(guarded[target]).ravel()[0])
        out["targets"][target] = {
            "raw": raw_value,
            "guarded": guard_value,
            "shift": guard_value - raw_value,
            "guarded_axis": target in physics_guard.GUARDED_TARGETS,
        }
    out["note"] = (
        "guarded 값은 dose 축 등장성 투영 결과입니다. "
        "TCAD 격자 재현 오차는 소폭 늘어나지만 dose 단조성이 보장됩니다."
    )
    return out


@app.get("/api/regional")
def regional_fit() -> dict[str, Any]:
    """Chungbuk semiconductor industry composition, aggregated from open data."""
    return chungbuk.deployment_case()



@app.post("/api/insight/llm")
def insight_llm(req: NarrateRequest) -> dict[str, Any]:
    return llm.narrate(req.evidence)


# --------------------------------------------------------------------------- #
# static frontend
# --------------------------------------------------------------------------- #
@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(SurrogateNotTrained)
def _not_trained_handler(_request, exc: SurrogateNotTrained) -> JSONResponse:  # noqa: ANN001
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

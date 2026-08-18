"""End-to-end verification of the AI Semiconductor Process Optimizer.

Run:
    python -m pytest tests -v          (if pytest installed)
    python tests/test_pipeline.py      (standalone, no pytest required)

Covers: dataset integrity, leakage policy, surrogate prediction, derived
delta_xj identity, extrapolation detection, XAI axioms, optimizer behaviour
and every REST endpoint.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from core import (  # noqa: E402
    chungbuk,
    config,
    dataset,
    doe_audit,
    explain,
    optimize,
    physics_guard,
    robust,
    surrogate,
    variation,
    whatif,
)

BASE = dict(dose_cm2=1.5e15, energy_keV=20.0, anneal_temp_C=1000.0, anneal_time_sec=25.0)
# core modules take positional (dose, energy, temp, time_s)
ARGS = tuple(BASE.values())


# --------------------------------------------------------------------- data
def test_dataset_shape_and_integrity():
    v = dataset.validate_dataset()
    assert v["n_rows"] == 1000
    assert v["n_columns"] == 9
    assert v["total_missing"] == 0
    assert v["duplicate_rows"] == 0
    assert v["duplicate_conditions"] == 0
    assert v["doe"]["is_full_factorial"] is True
    assert v["doe"]["expected_full_factorial"] == 1000
    assert v["delta_identity"]["holds"] is True


def test_doe_levels_match_design():
    lv = dataset.doe_levels()
    assert lv["dose_cm2"] == [5e14, 1e15, 1.5e15, 2e15, 2.5e15]
    assert lv["energy_keV"] == [10.0, 12.5, 15.0, 17.5, 20.0, 30.0, 40.0, 50.0]
    assert lv["anneal_temp_C"] == [900.0, 950.0, 1000.0, 1050.0, 1100.0]
    assert lv["anneal_time_sec"] == [5.0, 15.0, 25.0, 35.0, 45.0]


def test_no_output_column_can_become_a_feature():
    """Leakage guard: the feature matrix must contain exactly the 4 inputs."""
    df = dataset.load_dataset()
    X = dataset.build_features(df)
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert len(X.columns) == 4
    for forbidden in config.FORBIDDEN_AS_FEATURE:
        assert forbidden not in X.columns
    # log10(dose) must be a pure transform of dose only
    assert np.allclose(X["log10_dose"], np.log10(df["dose_cm2"]))


# ---------------------------------------------------------------- surrogate
def test_model_artifact_exists():
    assert surrogate.is_trained(), "run `python train_model.py` first"


def test_prediction_is_finite_and_in_physical_range():
    p = surrogate.predict_raw(*ARGS)
    tb = dataset.target_bounds()
    for t in ("xj_implant_um", "xj_final_um", "rsh_final_ohm_sq"):
        val = float(p[t][0])
        assert np.isfinite(val)
        span = tb[t]["max"] - tb[t]["min"]
        assert tb[t]["min"] - 0.25 * span <= val <= tb[t]["max"] + 0.25 * span, t


def test_delta_is_derived_not_modelled():
    p = surrogate.predict_raw(*ARGS)
    assert np.isclose(
        float(p["delta_xj_um"][0]), float(p["xj_final_um"][0]) - float(p["xj_implant_um"][0])
    )
    res = surrogate.delta_resolution()
    assert res["resolution_limit"] > 0


def test_surrogate_reproduces_real_tcad_runs():
    """The deployed models must track the CSV they were trained on."""
    df = dataset.load_dataset()
    p = surrogate.predict_raw(
        df["dose_cm2"], df["energy_keV"], df["anneal_temp_C"], df["anneal_time_sec"]
    )
    for t in ("xj_final_um", "rsh_final_ohm_sq"):
        y = df[t].to_numpy(dtype=float)
        r2 = 1 - np.sum((p[t] - y) ** 2) / np.sum((y - y.mean()) ** 2)
        assert r2 > 0.99, f"{t} in-sample R2 = {r2}"


def test_extrapolation_detection():
    assert surrogate.extrapolation_report(*ARGS)["level"] == "validated_doe_point"
    assert surrogate.extrapolation_report(1.2e15, 22, 1010, 26)["level"] == "interpolation"
    r = surrogate.extrapolation_report(5e15, 20, 1000, 25)
    assert r["level"] == "extrapolation"
    assert r["is_extrapolation"] is True
    assert "dose_cm2" in r["outside_parameters"]


def test_nearest_doe_run_exact_match():
    n = surrogate.nearest_doe_run(*ARGS)
    assert n["exact_match"] is True
    assert n["distance"] < 1e-9


# ---------------------------------------------------------------------- XAI
def test_shap_efficiency_axiom():
    """sum(phi) must equal f(x) - E[f(X)] for exact Shapley values."""
    e = explain.local_explanation(*ARGS)
    for t in config.MODEL_TARGETS:
        block = e[t]
        total = sum(s["value"] for s in block["shap"])
        assert np.isclose(total, block["prediction"] - block["base_value"], atol=1e-8), t
        assert abs(block["efficiency_residual"]) < 1e-8


def test_xj_implant_ignores_anneal_parameters():
    """xj_implant is a function of (dose, energy) only in the source data,
    so its attribution to anneal parameters must be ~0."""
    e = explain.local_explanation(*ARGS)
    contrib = {s["feature"]: abs(s["value"]) for s in e["xj_implant_um"]["shap"]}
    assert contrib["anneal_temp_C"] < 1e-6
    assert contrib["anneal_time_sec"] < 1e-6


def test_global_explanation_shape():
    g = explain.global_explanation()
    for t in config.MODEL_TARGETS:
        assert len(g["targets"][t]["permutation"]) == 4
        assert g["targets"][t]["permutation"][0]["rank"] == 1


# ---------------------------------------------------------------- what-if
def test_sweep_returns_matching_lengths_and_reference():
    s = whatif.sweep("anneal_temp_C", *ARGS, n_points=31)
    assert len(s["x"]) == 31
    assert len(s["curves"]["rsh_final_ohm_sq"]) == 31
    # base condition is on the DOE grid -> real TCAD runs must be available
    assert s["doe_reference"]["available"] is True
    assert s["doe_reference"]["n_points"] == 5


def test_sweep_rejects_unknown_parameter():
    try:
        whatif.sweep("bogus", *ARGS)
    except ValueError:
        return
    raise AssertionError("unknown sweep parameter must raise ValueError")


# -------------------------------------------------------------- optimizer
def test_doe_mode_returns_real_runs():
    r = optimize.optimize(0.25, 0.01, mode="doe", top_k=10)
    assert r["mode"] == "doe"
    assert r["verification_warning"] is None
    assert r["search"]["candidates_evaluated"] == 1000
    assert len(r["recipes"]) == 10
    df = dataset.load_dataset().set_index("run_id")
    for rec in r["recipes"]:
        row = df.loc[rec["run_id"]]
        # values reported in MODE A are the measured TCAD numbers
        assert np.isclose(rec["predicted"]["xj_final_um"], row["xj_final_um"])
        assert np.isclose(rec["predicted"]["rsh_final_ohm_sq"], row["rsh_final_ohm_sq"])
        assert np.isclose(rec["recipe"]["anneal_temp_C"], row["anneal_temp_C"])


def test_ai_mode_warns_and_stays_in_bounds():
    r = optimize.optimize(0.25, 0.01, mode="ai", top_k=5)
    assert "verification required" in r["verification_warning"]
    b = dataset.input_bounds()
    for rec in r["recipes"]:
        for k, bound in b.items():
            assert bound["min"] - 1e-6 <= rec["recipe"][k] <= bound["max"] + 1e-6, k
    assert r["search"]["candidates_evaluated"] > 1000


def test_recipes_are_ranked_and_feasible_first():
    r = optimize.optimize(0.24, 0.02, mode="doe", top_k=10)
    scores = [x["score"] for x in r["recipes"] if x["feasible"]]
    assert scores == sorted(scores)
    assert all(x["xj_error_um"] <= 0.02 + 1e-12 for x in r["recipes"] if x["feasible"])


def test_rsh_constraint_is_respected():
    r = optimize.optimize(0.25, 0.02, rsh_mode="constraint", rsh_max=60.0, mode="doe", top_k=10)
    for rec in r["recipes"]:
        if rec["feasible"]:
            assert rec["predicted"]["rsh_final_ohm_sq"] <= 60.0


def test_weight_shifts_the_solution():
    """Weighting Rsh harder must not produce a higher-Rsh rank-1 recipe."""
    a = optimize.optimize(0.25, 0.03, w_xj=1.0, w_rsh=0.0, mode="doe", top_k=1)
    b = optimize.optimize(0.25, 0.03, w_xj=0.0, w_rsh=1.0, mode="doe", top_k=1)
    assert b["recipes"][0]["predicted"]["rsh_final_ohm_sq"] <= a["recipes"][0]["predicted"]["rsh_final_ohm_sq"]


def test_pareto_front_is_non_dominated():
    r = optimize.optimize(0.25, 0.05, mode="doe", top_k=5)
    pts = [(p["xj_error_um"], p["rsh_final_ohm_sq"]) for p in r["pareto"]]
    assert len(pts) >= 2
    for i, (e1, s1) in enumerate(pts):
        for j, (e2, s2) in enumerate(pts):
            if i == j:
                continue
            assert not (e2 <= e1 and s2 <= s1 and (e2 < e1 or s2 < s1)), "dominated point on front"


def test_lock_constrains_the_search():
    r = optimize.optimize(0.25, 0.05, mode="doe", top_k=5, locks={"energy_keV": 20.0})
    for rec in r["recipes"]:
        assert np.isclose(rec["recipe"]["energy_keV"], 20.0)


# ------------------------------------------------- real-metrology variation
def test_metrology_reference_is_measured_not_assumed():
    """Variation defaults must come from the real 17,000-row metrology file.

    The raw CSV is company-supplied and not redistributed, so a fresh clone
    validates the committed aggregate instead of recomputing it.
    """
    assert variation.available(), "no variation reference (CSV or aggregate JSON)"
    r = variation.analyze(force=variation.raw_available())
    s = r["source"]
    assert s["rows"] == 17000
    assert s["wafers"] == 1000
    assert s["recipes"] == 6
    assert s["points_per_wafer"] == 13
    # Real equipment scatter is small but non-zero; a value of 0 or a wild
    # number would mean the aggregation is broken.
    w2w = r["wafer_to_wafer"]["cv_pct"]["mean"]
    assert 0.05 < w2w < 10.0, w2w
    assert 0 < r["within_wafer"]["nu_pct"]["mean"] < 50
    assert 0 < r["overall"]["total_cv_pct"] < 20
    # The scope caveat must survive - it is what keeps the claim honest.
    assert "PECVD" in r["caveat_en"]
    assert "이온주입" in r["caveat_ko"]


# ------------------------------------------------------------- robustness
def test_tolerance_presets_are_ordered():
    tight = robust.TOLERANCE_PRESETS["tight"]["cv_pct"]
    typical = robust.TOLERANCE_PRESETS["typical"]["cv_pct"]
    conservative = robust.TOLERANCE_PRESETS["conservative"]["cv_pct"]
    for p in robust.PARAMS:
        assert tight[p] <= typical[p] <= conservative[p], p


def test_resolve_tolerances_custom_flag():
    base = robust.resolve_tolerances("typical")
    assert base["is_custom"] is False
    echo = robust.resolve_tolerances("typical", dict(base["cv_pct"]))
    assert echo["is_custom"] is False, "echoing preset values must not read as custom"
    changed = robust.resolve_tolerances("typical", {"dose_cm2": 3.3})
    assert changed["is_custom"] is True
    assert changed["cv_pct"]["dose_cm2"] == 3.3


def test_zero_scatter_gives_perfect_yield():
    """Sanity anchor: no input scatter must collapse to the nominal answer."""
    r = robust.robust_analysis(
        *ARGS, target_xj_um=0.2452, tolerance_um=0.004,
        cv_pct={p: 0.0 for p in robust.PARAMS}, n_samples=400,
    )
    assert r["distribution"]["xj_final_um"]["std"] < 1e-9
    assert r["yield"]["xj_pct"] == 100.0


def test_wider_tolerance_never_improves_yield():
    kw = dict(target_xj_um=0.2452, tolerance_um=0.004, n_samples=1500)
    tight = robust.robust_analysis(*ARGS, preset="tight", **kw)
    loose = robust.robust_analysis(*ARGS, preset="conservative", **kw)
    assert loose["distribution"]["xj_final_um"]["std"] >= tight["distribution"]["xj_final_um"]["std"]
    assert loose["yield"]["xj_pct"] <= tight["yield"]["xj_pct"] + 1e-9


def test_variance_decomposition_shares_sum_to_100():
    r = robust.robust_analysis(*ARGS, target_xj_um=0.2452, tolerance_um=0.004, n_samples=800)
    dec = r["variance_decomposition"]
    assert len(dec) == 4
    assert abs(sum(d["xj_share_pct"] for d in dec) - 100) < 1e-6
    assert abs(sum(d["rsh_share_pct"] for d in dec) - 100) < 1e-6
    # xj_final is dominated by implant energy in the source data; the
    # decomposition must reproduce that rather than spreading blame evenly.
    assert max(dec, key=lambda d: d["xj_share_pct"])["parameter"] == "energy_keV"


def test_boundary_clipping_is_reported():
    """A setpoint on the DOE edge must not silently under-report its scatter."""
    interior = robust.robust_analysis(
        *ARGS, target_xj_um=0.2452, tolerance_um=0.004, n_samples=800
    )
    assert interior["boundary_clipping"]["significant"] is False

    edge = robust.robust_analysis(
        2.5e15, 20.0, 1000.0, 25.0, target_xj_um=0.25, tolerance_um=0.004, n_samples=800
    )
    assert edge["boundary_clipping"]["significant"] is True
    assert edge["boundary_clipping"]["pct_by_parameter"]["dose_cm2"] > 10


def test_samples_stay_inside_training_envelope():
    """Monte-Carlo must never push the surrogate into extrapolation."""
    b = dataset.input_bounds()
    rng = np.random.default_rng(0)
    s, _ = robust._sample(
        {"dose_cm2": 2.5e15, "energy_keV": 50.0, "anneal_temp_C": 1100.0, "anneal_time_sec": 45.0},
        {p: 5.0 for p in robust.PARAMS}, 500, rng,
    )
    for p in robust.PARAMS:
        assert s[p].min() >= b[p]["min"] - 1e-9
        assert s[p].max() <= b[p]["max"] + 1e-9


def test_robust_rank_is_sorted_by_yield():
    o = optimize.optimize(0.25, 0.004, mode="doe", top_k=6)
    ranked = robust.robust_rank(
        [r["recipe"] for r in o["recipes"]], 0.25, 0.004,
        preset="conservative", n_samples=600,
    )
    ys = [r["yield_joint_pct"] for r in ranked]
    assert ys == sorted(ys, reverse=True)
    assert [r["robust_rank"] for r in ranked] == list(range(1, len(ranked) + 1))
    assert all(r["rank_shift"] == r["nominal_rank"] - r["robust_rank"] for r in ranked)


def test_model_uncertainty_is_positive_and_labelled():
    u = surrogate.model_uncertainty(*ARGS)
    for t in config.MODEL_TARGETS:
        assert u[t]["std"][0] >= 0
        assert u[t]["source"]
        assert u[t]["test_rmse"] > 0


# ------------------------------------------------------------------- API
def test_rest_api_endpoints():
    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)

    assert client.get("/api/health").json()["model_trained"] is True

    meta = client.get("/api/meta").json()
    assert meta["dataset"]["rows"] == 1000
    assert meta["model_trained"] is True
    assert set(meta["model"]["best_models"]) == set(config.MODEL_TARGETS)

    summary = client.get("/api/dataset/summary").json()
    assert summary["validation"]["n_rows"] == 1000
    assert len(summary["correlation"]["labels"]) == 8

    pts = client.get("/api/dataset/points").json()["columns"]
    assert len(pts["dose_cm2"]) == 1000

    pred = client.post("/api/predict", json=BASE).json()
    assert set(pred["prediction"]) == {
        "xj_implant_um", "xj_final_um", "delta_xj_um", "rsh_final_ohm_sq"
    }
    assert pred["extrapolation"]["level"] == "validated_doe_point"
    assert len(pred["insights"]) >= 4
    levels = {c["level"] for c in pred["insights"]}
    assert "DATA OBSERVATION" in levels
    assert "MODEL INTERPRETATION" in levels
    assert "ENGINEERING VERIFICATION REQUIRED" in levels

    wi = client.post("/api/whatif", json={**BASE, "parameter": "anneal_temp_C", "n_points": 21}).json()
    assert len(wi["x"]) == 21

    opt = client.post("/api/optimize", json={"target_xj_um": 0.25, "tolerance_um": 0.01, "mode": "doe"}).json()
    assert len(opt["recipes"]) == 10
    assert len(opt["insights"]) >= 2

    xai = client.get("/api/xai/global").json()
    assert len(xai["targets"]["rsh_final_ohm_sq"]["permutation"]) == 4

    val = client.get("/api/validation").json()
    assert val["targets"]["xj_final_um"]["n_train"] + val["targets"]["xj_final_um"]["n_test"] == 1000
    assert "Fab data" in val["disclaimer_en"]

    assert client.get("/").status_code == 200
    assert client.get("/static/js/core.js").status_code == 200
    assert client.get("/static/vendor/chart.umd.js").status_code == 200
    assert client.get("/static/js/robust.js").status_code == 200

    var = client.get("/api/variation").json()
    assert var["available"] is True
    assert var["source"]["rows"] == 17000
    assert set(var["presets"]) == {"tight", "typical", "conservative"}

    rb = client.post(
        "/api/robust",
        json={**BASE, "target_xj_um": 0.2452, "tolerance_um": 0.004,
              "rsh_max": 70, "preset": "typical", "n_samples": 600},
    ).json()
    assert 0 <= rb["yield"]["joint_pct"] <= 100
    assert rb["cpk"]["xj"] is not None
    assert len(rb["variance_decomposition"]) == 4
    assert len(rb["distribution"]["xj_final_um"]["histogram"]["counts"]) > 5
    assert set(rb["model_uncertainty"]) == set(config.MODEL_TARGETS)

    opt_r = client.post(
        "/api/optimize",
        json={"target_xj_um": 0.25, "tolerance_um": 0.004, "mode": "doe",
              "top_k": 5, "robust_rerank": True, "robust_preset": "conservative"},
    ).json()
    assert len(opt_r["robust"]["ranking"]) == 5
    assert opt_r["robust"]["tolerances"]["preset"] == "conservative"


def test_api_rejects_invalid_input():
    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)
    r = client.post("/api/predict", json={**BASE, "dose_cm2": -1})
    assert r.status_code == 422


# ------------------------------------------------------- physics guard / DOE
def test_dose_is_the_only_axis_that_breaks_monotonicity():
    """Ten of eleven asserted (target, axis) pairs are already ordered."""
    measured = physics_guard.measure_monotonicity()
    offenders = {
        (target, axis)
        for target, axes in measured.items()
        for axis, stat in axes.items()
        if stat["asserted"] and stat["violation_rate"] > 0.05
    }
    assert offenders == {("xj_implant_um", "dose_cm2"), ("xj_final_um", "dose_cm2")}
    # Rsh falls with every input, exactly as implant/anneal physics requires.
    for axis in ("dose_cm2", "anneal_temp_C", "anneal_time_sec"):
        assert measured["rsh_final_ohm_sq"][axis]["violations"] == 0


def test_dose_artifact_is_systematic_not_random():
    """A per-level offset repeating across energies is a simulator artifact."""
    report = doe_audit.audit_monotonicity(dataset.load_dataset())
    assert not report["monotone_everywhere"]
    assert report["shape_reproducibility_corr"]["mean"] > 0.85


def test_guard_removes_every_dose_violation():
    lattice = physics_guard.get_lattice()
    for target in physics_guard.GUARDED_TARGETS:
        sign = physics_guard.MONOTONE_EXPECTATION[target]["dose_cm2"]
        assert sign > 0
        raw = np.diff(lattice.raw_values[target], axis=0)
        fixed = np.diff(lattice.values[target], axis=0)
        assert (raw < 0).sum() > 0
        assert (fixed < -1e-12).sum() == 0


def test_guard_cost_stays_below_the_models_own_error():
    """The constraint must not cost more than the surrogate's honest error."""
    report = physics_guard.audit()
    for stats in report["targets"].values():
        assert stats["lattice_violations_after"] == 0
        assert stats["cost_vs_group_cv_mae"] is not None
        assert stats["cost_vs_group_cv_mae"] < 0.5


def test_guard_preserves_the_delta_identity():
    guarded = physics_guard.predict_guarded(*ARGS)
    assert np.allclose(
        guarded["delta_xj_um"], guarded["xj_final_um"] - guarded["xj_implant_um"]
    )


def test_guarded_prediction_is_monotone_in_dose():
    bounds = dataset.input_bounds()
    doses = np.linspace(bounds["dose_cm2"]["min"], bounds["dose_cm2"]["max"], 25)
    guarded = physics_guard.predict_guarded(
        doses, np.full(25, 20.0), np.full(25, 1000.0), np.full(25, 25.0)
    )
    xj = guarded["xj_final_um"]
    assert np.all(np.isfinite(xj))
    assert np.all(np.diff(xj) >= -1e-12)


# ------------------------------------------------------------ regional case
def test_regional_profile_is_measured_from_the_register():
    profile = chungbuk.regional_profile()
    frame = chungbuk.load_firms()
    assert profile["total_firms"] == len(frame)
    assert sum(c["firms"] for c in profile["by_city"]) == len(frame)
    assert (
        sum(s["firms"] for s in profile["core_segments"])
        == profile["core_semiconductor_firms"]
    )


def test_regional_case_claims_no_demand():
    case = chungbuk.deployment_case()
    assert "수요를 추정하지 않았습니다" in case["caveat"]
    assert case["profile"]["core_semiconductor_firms"] > 0


def test_audit_and_regional_endpoints():
    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)

    guard = client.get("/api/physics/guard")
    assert guard.status_code == 200
    for stats in guard.json()["targets"].values():
        assert stats["lattice_violations_after"] == 0

    regional = client.get("/api/regional")
    assert regional.status_code == 200
    assert regional.json()["profile"]["total_firms"] == len(chungbuk.load_firms())

    compare = client.post("/api/physics/compare", json=BASE)
    assert compare.status_code == 200
    targets = compare.json()["targets"]
    assert targets["xj_final_um"]["guarded_axis"] is True
    assert abs(
        targets["delta_xj_um"]["guarded"]
        - (targets["xj_final_um"]["guarded"] - targets["xj_implant_um"]["guarded"])
    ) < 1e-12

    audit_resp = client.get("/api/doe/audit")
    assert audit_resp.status_code in (200, 503)
    if audit_resp.status_code == 200:
        assert audit_resp.json()["dataset"]["full_factorial"] is True


# ------------------------------------------------------------------ runner
def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

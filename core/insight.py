"""AI Process Insight generator.

Every sentence produced here is traceable to a number that was actually
computed, and each is tagged with one of three evidence levels:

  DATA OBSERVATION              - measured directly on implant_anneal_1000.csv
  MODEL INTERPRETATION          - derived from the trained surrogate / XAI
  ENGINEERING VERIFICATION REQ. - explicit caveat, never a physics claim

The generator never asserts a semiconductor physics mechanism that the data
does not show. It reports observed trends and attribution, and defers causal
interpretation to the engineer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from .config import FEATURE_LABELS, FEATURE_LABELS_KO, INPUT_META, TARGET_META
from .dataset import load_dataset, marginal_effects

FEATURE_TO_INPUT = {
    "log10_dose": "dose_cm2",
    "energy_keV": "energy_keV",
    "anneal_temp_C": "anneal_temp_C",
    "anneal_time_sec": "anneal_time_sec",
}

LEVEL_DATA = "DATA OBSERVATION"
LEVEL_MODEL = "MODEL INTERPRETATION"
LEVEL_VERIFY = "ENGINEERING VERIFICATION REQUIRED"


def _fmt(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.{digits}g}"
    return f"{value:.{digits}g}"


def data_observations(targets: tuple[str, ...] = ("rsh_final_ohm_sq", "xj_final_um")) -> list[dict[str, Any]]:
    """Balanced full-factorial main effects + rank correlation, from the CSV."""
    df = load_dataset()
    eff = marginal_effects(df)
    items: list[dict[str, Any]] = []
    for target in targets:
        tmeta = TARGET_META[target]
        for param, block in eff.items():
            levels = np.asarray(block["levels"], dtype=float)
            means = np.asarray(block[target], dtype=float)
            span = float(means.max() - means.min())
            rel = span / float(np.mean(means)) * 100.0 if np.mean(means) != 0 else 0.0
            rho, pval = spearmanr(df[param].to_numpy(dtype=float), df[target].to_numpy(dtype=float))
            direction_ko = "감소" if means[-1] < means[0] else ("증가" if means[-1] > means[0] else "유지")
            items.append(
                {
                    "level": LEVEL_DATA,
                    "target": target,
                    "parameter": param,
                    "effect_span": span,
                    "effect_span_pct": rel,
                    "spearman_rho": float(rho),
                    "spearman_p": float(pval),
                    "text_ko": (
                        f"TCAD DOE 1,000 run 전체에서 {INPUT_META[param]['label_ko']}"
                        f"({_fmt(levels[0])} → {_fmt(levels[-1])} {INPUT_META[param]['unit']}) 변화에 따라 "
                        f"{tmeta['label_ko']} 평균은 {_fmt(means[0])} → {_fmt(means[-1])} {tmeta['unit']}로 "
                        f"{direction_ko}하는 경향이 관측되었습니다 (Spearman ρ = {rho:.3f})."
                    ),
                    "text_en": (
                        f"Across all 1,000 TCAD runs, sweeping {INPUT_META[param]['label']} "
                        f"({_fmt(levels[0])} → {_fmt(levels[-1])} {INPUT_META[param]['unit']}) moves the mean "
                        f"{tmeta['label']} from {_fmt(means[0])} to {_fmt(means[-1])} {tmeta['unit']} "
                        f"(Spearman rho = {rho:.3f})."
                    ),
                }
            )
    items.sort(key=lambda d: -abs(d["effect_span_pct"]))
    return items


def build_insights(
    prediction: dict[str, Any],
    explanation: dict[str, Any],
    extrapolation: dict[str, Any],
    model_metrics: dict[str, Any],
    nearest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Insight cards for one operating point."""
    cards: list[dict[str, Any]] = []
    obs = data_observations()

    # ---------- DATA OBSERVATION (top main effect per main target) ----------
    for target in ("rsh_final_ohm_sq", "xj_final_um"):
        best = max((o for o in obs if o["target"] == target), key=lambda d: abs(d["effect_span_pct"]))
        cards.append(
            {
                "level": LEVEL_DATA,
                "title_ko": f"{TARGET_META[target]['label_ko']} 주효과 (DOE 전체)",
                "title_en": f"{TARGET_META[target]['label']} main effect (full DOE)",
                "text_ko": best["text_ko"],
                "text_en": best["text_en"],
                "evidence": {
                    "source": "implant_anneal_1000.csv",
                    "method": "balanced full-factorial marginal mean + Spearman rank correlation",
                    "spearman_rho": best["spearman_rho"],
                    "effect_span": best["effect_span"],
                },
            }
        )

    # ---------- MODEL INTERPRETATION (local SHAP at this operating point) ----
    for target in ("xj_final_um", "rsh_final_ohm_sq"):
        block = explanation.get(target)
        if not block:
            continue
        shap_items = block["shap"]
        top = shap_items[0]
        second = shap_items[1] if len(shap_items) > 1 else None
        sens = {s["feature"]: s for s in block["sensitivity"]}
        top_sens = sens.get(top["feature"], {})
        direction_ko = (
            "증가" if top_sens.get("signed", 0) > 0 else ("감소" if top_sens.get("signed", 0) < 0 else "변화 없음")
        )
        tmeta = TARGET_META[target]
        text_ko = (
            f"현재 입력 조건에서 {tmeta['label_ko']} 예측값 "
            f"{_fmt(block['prediction'])} {tmeta['unit']}에 대해, 데이터 평균(base value "
            f"{_fmt(block['base_value'])})으로부터의 편차를 가장 크게 설명하는 변수는 "
            f"{FEATURE_LABELS_KO[top['feature']]}입니다 (SHAP φ = {_fmt(top['value'])}, "
            f"기여 비중 {top['share_pct']:.1f}%)."
        )
        if second:
            text_ko += f" 다음은 {FEATURE_LABELS_KO[second['feature']]} ({second['share_pct']:.1f}%)입니다."
        text_ko += (
            f" 이 조건에서 {FEATURE_LABELS_KO[top['feature']]}만 학습 범위 전체로 변화시키면 "
            f"{tmeta['label_ko']}은 약 {_fmt(top_sens.get('value', 0.0))} {tmeta['unit']} 폭으로 움직이며, "
            f"방향은 {direction_ko}입니다 (surrogate model 기준)."
        )
        text_en = (
            f"For the current condition the surrogate predicts {tmeta['label']} = "
            f"{_fmt(block['prediction'])} {tmeta['unit']}. The dominant contributor to the deviation from the "
            f"dataset mean ({_fmt(block['base_value'])}) is {FEATURE_LABELS[top['feature']]} "
            f"(SHAP phi = {_fmt(top['value'])}, {top['share_pct']:.1f}% of total attribution). "
            f"Sweeping it across the full training range moves {tmeta['label']} by "
            f"{_fmt(top_sens.get('value', 0.0))} {tmeta['unit']}."
        )
        cards.append(
            {
                "level": LEVEL_MODEL,
                "title_ko": f"{tmeta['label_ko']} 국소 기여도 해석",
                "title_en": f"{tmeta['label']} local attribution",
                "text_ko": text_ko,
                "text_en": text_en,
                "evidence": {
                    "method": block["method"],
                    "efficiency_residual": block["efficiency_residual"],
                    "top_feature": top["feature"],
                },
            }
        )

    # ---------- ENGINEERING VERIFICATION REQUIRED ----------
    xj_metrics = model_metrics.get("xj_final_um", {}).get("test", {})
    rsh_metrics = model_metrics.get("rsh_final_ohm_sq", {}).get("test", {})
    verify_ko = (
        f"본 예측은 TCAD 시뮬레이션 DOE 데이터로 학습한 surrogate model 결과입니다. "
        f"Hold-out test 기준 오차는 Xj Final MAE {_fmt(xj_metrics.get('mae', float('nan')))} um, "
        f"Rsh MAE {_fmt(rsh_metrics.get('mae', float('nan')))} ohm/sq 이며, "
        f"이 값은 실제 Fab 공정 산포가 아닌 시뮬레이션 재현 오차입니다."
    )
    verify_en = (
        "This prediction comes from a surrogate trained on TCAD simulation DOE data. "
        f"Held-out test error is Xj Final MAE {_fmt(xj_metrics.get('mae', float('nan')))} um and "
        f"Rsh MAE {_fmt(rsh_metrics.get('mae', float('nan')))} ohm/sq; these represent simulation "
        "reproduction error, not Fab process variation."
    )
    if extrapolation["level"] == "extrapolation":
        verify_ko += " 추가로 입력 조건이 학습 범위를 벗어나 외삽 상태이므로 결과를 의사결정에 직접 사용하지 마십시오."
        verify_en += " In addition, the input is outside the training envelope (extrapolation)."
    elif extrapolation["level"] == "interpolation":
        verify_ko += " 입력 조건은 DOE 격자 사이의 보간 영역이므로 TCAD 재시뮬레이션으로 확인하는 것을 권장합니다."
        verify_en += " The input sits between DOE grid points, so a confirming TCAD run is recommended."
    if nearest and not nearest.get("exact_match"):
        run = nearest["run"]
        verify_ko += (
            f" 가장 가까운 실제 TCAD run은 run_id {int(run['run_id'])} "
            f"(Dose {_fmt(run['dose_cm2'])} cm^-2, {_fmt(run['energy_keV'])} keV, "
            f"{_fmt(run['anneal_temp_C'])} °C, {_fmt(run['anneal_time_sec'])} sec, "
            f"Xj Final {_fmt(run['xj_final_um'])} um, Rsh {_fmt(run['rsh_final_ohm_sq'])} ohm/sq)입니다."
        )
    cards.append(
        {
            "level": LEVEL_VERIFY,
            "title_ko": "엔지니어링 검증 필요 사항",
            "title_en": "Engineering verification required",
            "text_ko": verify_ko,
            "text_en": verify_en,
            "evidence": {
                "extrapolation_level": extrapolation["level"],
                "nearest_doe_distance": (nearest or {}).get("distance"),
            },
        }
    )
    return cards


def optimizer_insights(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Insight cards for an optimizer run."""
    cards: list[dict[str, Any]] = []
    obj = result["objective"]
    search = result["search"]
    recipes = result["recipes"]
    if not recipes:
        return cards
    best = recipes[0]
    feasible_n = search["feasible_count"]

    cards.append(
        {
            "level": LEVEL_DATA if result["mode"] == "doe" else LEVEL_MODEL,
            "title_ko": "탐색 결과 요약",
            "title_en": "Search summary",
            "text_ko": (
                f"{search['candidates_evaluated']:,}개 후보 조건을 평가하여 "
                f"목표 Xj {obj['target_xj_um']} ± {obj['tolerance_um']} um 조건을 만족하는 후보 "
                f"{feasible_n:,}건을 확인했습니다. 1순위 조건은 Dose {_fmt(best['recipe']['dose_cm2'])} cm^-2, "
                f"{_fmt(best['recipe']['energy_keV'])} keV, {_fmt(best['recipe']['anneal_temp_C'])} °C, "
                f"{_fmt(best['recipe']['anneal_time_sec'])} sec 이며 Xj {_fmt(best['predicted']['xj_final_um'])} um, "
                f"Rsh {_fmt(best['predicted']['rsh_final_ohm_sq'])} ohm/sq 입니다."
            ),
            "text_en": (
                f"{search['candidates_evaluated']:,} candidate conditions were evaluated; "
                f"{feasible_n:,} satisfy Xj = {obj['target_xj_um']} +/- {obj['tolerance_um']} um. "
                f"Rank 1 recipe: dose {_fmt(best['recipe']['dose_cm2'])} cm^-2, "
                f"{_fmt(best['recipe']['energy_keV'])} keV, {_fmt(best['recipe']['anneal_temp_C'])} degC, "
                f"{_fmt(best['recipe']['anneal_time_sec'])} sec."
            ),
            "evidence": {"mode": result["mode_label"], "score_formula": obj["formula"]},
        }
    )

    pareto = result.get("pareto", [])
    if len(pareto) >= 2:
        lo, hi = pareto[0], pareto[-1]
        cards.append(
            {
                "level": LEVEL_MODEL,
                "title_ko": "Trade-off (Pareto) 관측",
                "title_en": "Observed trade-off (Pareto front)",
                "text_ko": (
                    f"Pareto front 상에서 Xj 오차 {_fmt(lo['xj_error_um'])} um 조건의 Rsh는 "
                    f"{_fmt(lo['rsh_final_ohm_sq'])} ohm/sq, Xj 오차 {_fmt(hi['xj_error_um'])} um 조건의 Rsh는 "
                    f"{_fmt(hi['rsh_final_ohm_sq'])} ohm/sq로, 접합깊이 정확도와 면저항 사이의 trade-off가 "
                    f"{len(pareto)}개 비지배 후보로 나타납니다. 단일 최적해가 아니라 공정 우선순위에 따른 선택이 필요합니다."
                ),
                "text_en": (
                    f"The non-dominated set contains {len(pareto)} candidates; Xj error "
                    f"{_fmt(lo['xj_error_um'])} um pairs with Rsh {_fmt(lo['rsh_final_ohm_sq'])} ohm/sq while "
                    f"Xj error {_fmt(hi['xj_error_um'])} um pairs with Rsh {_fmt(hi['rsh_final_ohm_sq'])} ohm/sq."
                ),
                "evidence": {"pareto_points": len(pareto)},
            }
        )

    if result["mode"] == "doe":
        checks = [r.get("surrogate_check") for r in recipes if r.get("surrogate_check")]
        if checks:
            xj_err = float(np.mean([c["xj_abs_error_um"] for c in checks]))
            rsh_err = float(np.mean([c["rsh_abs_error_ohm_sq"] for c in checks]))
            cards.append(
                {
                    "level": LEVEL_VERIFY,
                    "title_ko": "검증 상태",
                    "title_en": "Verification status",
                    "text_ko": (
                        "MODE A 결과는 실제 TCAD 시뮬레이션 값이므로 추가 보간 오차가 없습니다. "
                        f"참고로 동일 조건에서 surrogate model의 평균 절대 오차는 Xj {_fmt(xj_err)} um, "
                        f"Rsh {_fmt(rsh_err)} ohm/sq 입니다. 실제 양산 적용은 Fab split 검증이 필요합니다."
                    ),
                    "text_en": (
                        "MODE A recipes are actual TCAD simulation results, so no interpolation error is "
                        f"involved. On the same conditions the surrogate deviates by {_fmt(xj_err)} um (Xj) "
                        f"and {_fmt(rsh_err)} ohm/sq (Rsh) on average. Fab split verification is still required."
                    ),
                    "evidence": {"surrogate_mean_abs_error": {"xj_final_um": xj_err, "rsh_final_ohm_sq": rsh_err}},
                }
            )
    else:
        cards.append(
            {
                "level": LEVEL_VERIFY,
                "title_ko": "검증 상태",
                "title_en": "Verification status",
                "text_ko": (
                    "MODE B 결과는 학습된 surrogate model의 보간 예측입니다. "
                    "AI interpolation result — TCAD/Fab verification required. "
                    "추천 조건은 TCAD 재시뮬레이션으로 확인한 뒤 split 실험 후보로 사용하십시오."
                ),
                "text_en": (
                    "MODE B recipes are surrogate interpolations. "
                    "AI interpolation result — TCAD/Fab verification required."
                ),
                "evidence": {"mode": result["mode_label"]},
            }
        )
    return cards

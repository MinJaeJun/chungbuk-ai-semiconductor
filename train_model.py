"""Train the TCAD surrogate models.

Usage:
    python train_model.py
    python train_model.py --quick      # skip global SHAP (faster)

Produces:
    models/surrogate_bundle.joblib   deployment models + full report
    outputs/training_report.json     human/UI readable metrics
    outputs/leaderboard_<target>.csv model comparison tables
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn

from core.config import (
    ARTIFACT_PATH,
    DISCLAIMER_EN,
    DISCLAIMER_KO,
    MODEL_TARGETS,
    OUTPUT_DIR,
    REPORT_PATH,
    ensure_dirs,
)
from core.dataset import (
    correlation_matrix,
    load_dataset,
    marginal_effects,
    validate_dataset,
)
from core.explain import global_shap_importance
from core.modeling import OPTIONAL_LIBS, train_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TCAD surrogate models")
    parser.add_argument("--quick", action="store_true", help="skip global SHAP computation")
    parser.add_argument("--shap-sample", type=int, default=100)
    parser.add_argument("--shap-background", type=int, default=100)
    args = parser.parse_args()

    ensure_dirs()
    t_start = time.perf_counter()

    print("=" * 78)
    print(" AI Semiconductor Process Optimizer - Surrogate Model Training")
    print("=" * 78)

    df = load_dataset()
    validation = validate_dataset(df)
    print(f"\n[DATA] {validation['csv_path']}")
    print(f"  rows={validation['n_rows']}  cols={validation['n_columns']}  "
          f"missing={validation['total_missing']}  duplicate_rows={validation['duplicate_rows']}")
    print(f"  DOE structure: {validation['doe']['structure']}  "
          f"full_factorial={validation['doe']['is_full_factorial']}")
    print(f"  identity {validation['delta_identity']['expression']} -> "
          f"max|residual| = {validation['delta_identity']['max_abs_residual']:.3g}")
    print(f"  features used: {validation['leakage_policy']['features_used']}")
    print(f"  excluded (leakage guard): {validation['leakage_policy']['excluded_from_features']}")
    print(f"  optional libs: {OPTIONAL_LIBS}")

    results, delta_report = train_all(verbose=True)

    deploy_models = {t: results[t]["_deploy_model"] for t in MODEL_TARGETS}

    global_shap: dict[str, dict[str, float]] = {}
    if not args.quick:
        print("\n[XAI] computing global SHAP importance (exact Shapley, 2^4 coalitions)...")
        t0 = time.perf_counter()
        global_shap = global_shap_importance(
            deploy_models, n_sample=args.shap_sample, n_background=args.shap_background
        )
        print(f"  done in {time.perf_counter() - t0:.1f}s")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "optional_libraries": OPTIONAL_LIBS,
        },
        "dataset": validation,
        "correlation": correlation_matrix(df),
        "marginal_effects": marginal_effects(df),
        "targets": {
            t: {k: v for k, v in results[t].items() if not k.startswith("_")}
            for t in MODEL_TARGETS
        },
        "global_shap": global_shap,
        "derived_delta": delta_report,
        "derived_targets": {
            "delta_xj_um": "predicted xj_final_um - predicted xj_implant_um "
            "(dataset identity verified, max residual "
            f"{validation['delta_identity']['max_abs_residual']:.3g}); "
            "the model pair is chosen to minimise this derived quantity's hold-out RMSE"
        },
        "disclaimer_en": DISCLAIMER_EN,
        "disclaimer_ko": DISCLAIMER_KO,
        "training_seconds": None,
    }
    report["training_seconds"] = round(time.perf_counter() - t_start, 2)

    bundle = {
        "trained_at": report["generated_at"],
        "models": deploy_models,
        "report": report,
    }
    import joblib

    joblib.dump(bundle, ARTIFACT_PATH, compress=3)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    for t in MODEL_TARGETS:
        pd.DataFrame(results[t]["leaderboard"]).to_csv(
            OUTPUT_DIR / f"leaderboard_{t}.csv", index=False
        )

    print("\n" + "=" * 78)
    print(" RESULT SUMMARY (all values measured, none hard-coded)")
    print("=" * 78)
    for t in MODEL_TARGETS:
        m = results[t]["metrics"]
        print(f"\n {t}")
        print(f"   best model : {results[t]['best_model']}")
        print(f"   selection  : {results[t]['selection_metric']}")
        print(f"   GROUP CV (unseen dose x energy, {results[t]['n_groups']} groups)"
              f"  R2={m['group_cv']['r2']:.6f}  MAE={m['group_cv']['mae']:.6g}  RMSE={m['group_cv']['rmse']:.6g}")
        print(f"   RANDOM CV (5-fold, train)   R2={m['cv']['r2']:.6f}  MAE={m['cv']['mae']:.6g}  RMSE={m['cv']['rmse']:.6g}")
        print(f"   TEST (hold-out 20%)         R2={m['test']['r2']:.6f}  MAE={m['test']['mae']:.6g}  RMSE={m['test']['rmse']:.6g}")
        if global_shap:
            ranked = sorted(global_shap[t].items(), key=lambda kv: -kv[1])
            print("   global |SHAP| : " + ", ".join(f"{k}={v:.4g}" for k, v in ranked))

    d = delta_report["test"]
    print("\n delta_xj_um (DERIVED, not trained directly)")
    print(f"   formula    : pred(xj_final_um) - pred(xj_implant_um)")
    print(f"   pair       : implant={delta_report['selected_pair']['xj_implant_um']} | "
          f"final={delta_report['selected_pair']['xj_final_um']}")
    print(f"   TEST         R2={d['r2']:.6f}  MAE={d['mae']:.6g}  RMSE={d['rmse']:.6g}  "
          f"({delta_report['pairs_evaluated']} pairs evaluated)")

    print(f"\n artifact : {ARTIFACT_PATH}")
    print(f" report   : {REPORT_PATH}")
    print(f" elapsed  : {report['training_seconds']}s")
    print("\n " + DISCLAIMER_KO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

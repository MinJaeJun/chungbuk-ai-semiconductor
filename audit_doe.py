"""Runner for the DOE information-content audit.

    python audit_doe.py            # full audit (GP learning curve, ~10-20 min)
    python audit_doe.py --quick    # reduced learning curve, ~3-5 min

Writes outputs/doe_audit_report.json and prints a console summary.  Every
figure is measured from data/implant_anneal_1000.csv at run time.
"""

from __future__ import annotations

import argparse
import sys
import warnings

from sklearn.exceptions import ConvergenceWarning

from core import config
from core.doe_audit import AUDIT_TARGETS, run_audit, save_audit

# GP hyper-parameter optimisation routinely pins the noise term against its
# lower bound on this near-noiseless simulation grid.  That is expected here and
# does not affect the reported scores, so the notice is silenced to keep the
# audit summary readable.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _line(char: str = "=", width: int = 78) -> str:
    return char * width


def print_summary(payload: dict) -> None:
    data = payload["dataset"]
    print(_line())
    print(" DOE INFORMATION-CONTENT AUDIT")
    print(_line())
    print(f"[DATA] {data['csv']}  rows={data['rows']}")
    design = " x ".join(str(len(v)) for v in data["levels"].values())
    print(
        f"  design {design} = {data['design_product']}  "
        f"full_factorial={data['full_factorial']}  "
        f"duplicate_input_rows={data['duplicate_input_rows']}"
    )

    suf = payload["sufficiency"]
    print()
    print(f"[1] SUFFICIENCY  learning curve, R2 threshold {suf['r2_threshold']}")
    for target in AUDIT_TARGETS:
        print(f"  {target}")
        for point in suf["curve"][target]:
            print(
                f"    N={point['n_train']:4d}  R2={point['r2']:.6f}  "
                f"MAE={point['mae']:.6g}"
            )
        need = suf["runs_needed"][target]
        save = suf["compute_saving_pct"][target]
        if need is None:
            print(f"    -> threshold not reached below N={suf['total_runs']}")
        else:
            print(f"    -> N={need} reaches threshold ({save}% fewer runs)")
    both = suf["runs_needed_both_targets"]
    if both is not None:
        print(
            f"  BOTH targets: {both} of {suf['total_runs']} runs suffice  "
            f"-> {suf['compute_saving_pct_both_targets']}% TCAD compute saved"
        )

    mono = payload["monotonicity"]
    print()
    print("[2] MONOTONICITY  x_j_implant vs dose at fixed energy")
    print(
        f"  monotone_everywhere={mono['monotone_everywhere']}  "
        f"violations={mono['violation_count']} / "
        f"{len(mono['per_energy']) * (len(mono['grid']['doses_cm2']) - 1)} steps"
    )
    shape = mono["dose_shape_pct_of_mean"]
    print("  dose response, % deviation from each energy's own mean:")
    for dose, dev, std in zip(
        shape["dose_cm2"], shape["mean_dev_pct"], shape["std_dev_pct"]
    ):
        print(f"    dose={dose:.2e}  mean={dev:+7.3f}%  std={std:6.3f}%")
    corr = mono["shape_reproducibility_corr"]
    print(
        f"  shape reproducibility across energies: mean r={corr['mean']:.4f}  "
        f"min r={corr['min']:.4f}"
    )

    ext = payload["extrapolation"]
    print()
    print("[3] EXTRAPOLATION  interior box -> outer shell")
    print(
        f"  train={ext['n_train_interior']}  test={ext['n_test_outer_shell']}"
    )
    for target, stats in ext["targets"].items():
        print(
            f"  {target:<18} outer R2={stats['outer_shell_r2']:.4f}  "
            f"MAE={stats['outer_shell_mae']:.5g}  "
            f"(interior MAE={stats['interior_fit_mae']:.5g}, "
            f"x{stats['mae_inflation_factor']:.0f} inflation)"
        )

    th = payload["thermal_axis"]
    print()
    print(f"[4] THERMAL AXIS  Dt = t * exp(-Ea/kT), target={th['target']}")
    for row in th["ea_scan"]:
        print(f"    Ea={row['ea_ev']:4.1f} eV  collapse R2={row['collapse_r2']:.6f}")
    print(
        f"  best Ea={th['best_fit']['ea_ev']:.1f} eV -> "
        f"R2={th['best_fit']['collapse_r2']:.6f}"
    )
    print(
        f"  boron reference Ea={th['boron_reference']['ea_ev']} eV -> "
        f"R2={th['boron_reference']['collapse_r2']:.6f}"
    )
    print(
        f"  1-D Dt quadratic ({th['params']['dt_axis_quadratic']} params) vs "
        f"2-D (temp,time) quadratic ({th['params']['temp_time_quadratic']} params) "
        f"R2={th['baseline_2d_quadratic_r2']:.6f}"
    )
    print("  group CV (unseen dose x energy):")
    for name, per_target in th["group_cv"].items():
        for target, stats in per_target.items():
            print(
                f"    {name:<18} {target:<18} R2={stats['group_cv_r2']:.6f}  "
                f"MAE={stats['group_cv_mae']:.5g}"
            )

    print()
    print(_line())
    print(f" elapsed {payload['elapsed_sec']:.1f}s")
    print(_line())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="reduced learning curve for a fast check"
    )
    args = parser.parse_args(argv)

    if not config.CSV_PATH.exists():
        print(f"[ERROR] dataset not found: {config.CSV_PATH}", file=sys.stderr)
        return 1

    result = run_audit(quick=args.quick)
    payload = result.to_dict()
    print_summary(payload)
    path = save_audit(result)
    print(f"report : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

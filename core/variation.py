"""Real-metrology process variation reference.

Source: data/fab_thickness_profile_17000.csv
    17,000 real in-line thickness measurements (PECVD SiON), 40 lots x 25 wafers
    x 6 recipes, 13-point wafer map, 2024-11-06 ~ 2024-11-08.

WHY THIS FILE EXISTS
--------------------
The TCAD DOE is deterministic: every condition has exactly one answer and the
surrogate reproduces it to ~1e-8. A real line does not behave that way. Without
a variation model, any "optimal recipe" this platform recommends is a
single-point answer that ignores the thing process engineers actually care
about - whether the recipe still meets spec once real equipment scatter is
applied.

This module measures how large that scatter actually is on a REAL production
line, so the robustness analysis in core/robust.py is anchored to observed
data instead of an invented number.

HONEST SCOPE - READ BEFORE CITING
---------------------------------
The measured process here is PECVD SiON deposition, NOT ion implantation.
These numbers are therefore used ONLY as an order-of-magnitude reference for
choosing input tolerances, and never as an implant process model. The
robustness tool always lets the engineer override every tolerance, and the UI
labels the source process explicitly.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR

METROLOGY_CSV = DATA_DIR / "fab_thickness_profile_17000.csv"
VARIATION_JSON = OUTPUT_DIR / "variation_reference.json"

# The wafer map is carried by whichever measurement item has the most rows; the
# other item is a separate edge measurement. Detecting it beats hard-coding the
# supplier's item label: it keeps customer naming out of this repository and
# still works if the metrology export is renamed.
def _primary_item(df: pd.DataFrame) -> str:
    return str(df["item_id"].value_counts().idxmax())

_CACHE: dict[str, Any] = {}


def available() -> bool:
    """True when a variation reference can be served.

    The raw metrology CSV is company-supplied and is deliberately NOT committed
    (see .gitignore). Only the derived aggregate - percentages and counts, no
    per-wafer readings - is version-controlled, so a fresh clone still gets a
    working reference without redistributing the source measurements.
    """
    return METROLOGY_CSV.exists() or VARIATION_JSON.exists()


def raw_available() -> bool:
    """True only when the original measurement file is present locally."""
    return METROLOGY_CSV.exists()


def _load() -> pd.DataFrame:
    if "df" not in _CACHE:
        df = pd.read_csv(METROLOGY_CSV, encoding="utf-8-sig")
        df["date_time"] = pd.to_datetime(df["date_time"])
        _CACHE["df"] = df
    return _CACHE["df"]


def analyze(force: bool = False) -> dict[str, Any]:
    """Measure within-wafer / wafer-to-wafer / lot-to-lot variation."""
    if not force and VARIATION_JSON.exists():
        return json.loads(VARIATION_JSON.read_text(encoding="utf-8"))
    if not raw_available():
        raise FileNotFoundError(
            f"Raw metrology CSV not present ({METROLOGY_CSV.name}); it is not "
            "distributed with this repository. The committed aggregate at "
            f"{VARIATION_JSON.name} is used instead - delete it and supply the "
            "CSV to recompute."
        )

    df = _load()
    d = df[df["item_id"] == _primary_item(df)]

    # ---- wafer level: 13 points per wafer -------------------------------
    wafer = (
        d.groupby(["process_recipe", "lot_id", "wafer_id"])["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    wafer["nu_pct"] = (wafer["max"] - wafer["min"]) / (2 * wafer["mean"]) * 100
    wafer["cv_pct"] = wafer["std"] / wafer["mean"] * 100

    # ---- lot level: spread of wafer means inside one lot ----------------
    lot = (
        wafer.groupby(["process_recipe", "lot_id"])["mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    lot["w2w_cv_pct"] = lot["std"] / lot["mean"] * 100

    # ---- recipe level: spread of lot means inside one recipe ------------
    recipe = lot.groupby("process_recipe")["mean"].agg(["mean", "std", "count"]).reset_index()
    recipe["l2l_cv_pct"] = recipe["std"] / recipe["mean"] * 100

    def stats(series: pd.Series) -> dict[str, float]:
        s = series.dropna()
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p90": float(s.quantile(0.90)),
            "max": float(s.max()),
            "n": int(len(s)),
        }

    # ---- radial signature of the within-wafer map ------------------------
    pts = d.copy()
    pts["r"] = np.hypot(pts["x"], pts["y"])
    by_r = pts.groupby("r")["value"].agg(["mean", "std", "count"]).reset_index()
    grand = float(pts["value"].mean())
    by_r["dev_pct"] = (by_r["mean"] - grand) / grand * 100

    report = {
        "source": {
            "file": METROLOGY_CSV.name,
            "process": "PECVD SiON deposition (in-line thickness metrology)",
            "process_ko": "PECVD SiON 증착 · 인라인 두께 계측",
            "rows": int(len(df)),
            "rows_primary_item": int(len(d)),
            "lots": int(d["lot_id"].nunique()),
            "wafers": int(len(wafer)),
            "recipes": int(d["process_recipe"].nunique()),
            "points_per_wafer": int(wafer["count"].mode().iloc[0]),
            "date_from": str(df["date_time"].min()),
            "date_to": str(df["date_time"].max()),
            "unit": "angstrom",
        },
        "caveat_en": (
            "Measured on PECVD SiON deposition, a different process from ion "
            "implantation. Used only as an order-of-magnitude reference for "
            "selecting process input tolerances - never as an implant model."
        ),
        "caveat_ko": (
            "본 산포는 이온주입이 아닌 PECVD SiON 증착 공정의 실측값입니다. "
            "공정 입력 tolerance의 크기를 정할 때 참조 기준으로만 사용하며, "
            "이온주입 공정 모델로 사용하지 않습니다."
        ),
        "within_wafer": {
            "nu_pct": stats(wafer["nu_pct"]),
            "cv_pct": stats(wafer["cv_pct"]),
            "definition": "NU% = (max-min)/(2*mean) over the 13-point map; CV% = std/mean",
        },
        "wafer_to_wafer": {
            "cv_pct": stats(lot["w2w_cv_pct"]),
            "wafers_per_lot": int(lot["count"].mode().iloc[0]),
            "definition": "std/mean of wafer-mean thickness inside one lot",
        },
        "lot_to_lot": {
            "cv_pct": stats(recipe["l2l_cv_pct"]),
            "per_recipe": [
                {
                    "recipe": r["process_recipe"],
                    "lots": int(r["count"]),
                    "mean": float(r["mean"]),
                    "cv_pct": float(r["l2l_cv_pct"]) if np.isfinite(r["l2l_cv_pct"]) else None,
                }
                for _, r in recipe.iterrows()
            ],
            "definition": "std/mean of lot-mean thickness inside one recipe",
        },
        "overall": {
            "total_cv_pct": float(wafer["mean"].std() / wafer["mean"].mean() * 100),
            "definition": "std/mean over every wafer mean, all recipes pooled",
        },
        "radial_profile": [
            {
                "r_mm": float(row["r"]),
                "mean": float(row["mean"]),
                "dev_pct": float(row["dev_pct"]),
                "n": int(row["count"]),
            }
            for _, row in by_r.iterrows()
        ],
        "recipe_summary": [
            {
                "recipe": r,
                "wafers": int(g["mean"].count()),
                "mean": float(g["mean"].mean()),
                "std": float(g["mean"].std()),
                "within_wafer_nu_pct": float(g["nu_pct"].mean()),
            }
            for r, g in wafer.groupby("process_recipe")
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VARIATION_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def reference_cv_pct() -> dict[str, float]:
    """Headline numbers used as defaults / anchors elsewhere."""
    r = analyze()
    return {
        "within_wafer_nu_pct": r["within_wafer"]["nu_pct"]["mean"],
        "wafer_to_wafer_cv_pct": r["wafer_to_wafer"]["cv_pct"]["mean"],
        "wafer_to_wafer_cv_pct_p90": r["wafer_to_wafer"]["cv_pct"]["p90"],
        "lot_to_lot_cv_pct": r["lot_to_lot"]["cv_pct"]["mean"],
        "total_cv_pct": r["overall"]["total_cv_pct"],
    }

"""Build a static, zero-install demo of the dashboard for GitHub Pages.

The live app needs a Python backend to run the surrogate. GitHub Pages serves
files only, so this script pre-computes the model's answers on the 1,000 TCAD
DOE grid points and writes them next to a copy of the frontend. The browser
then reads JSON files instead of calling the API (see static/js/demo-shim.js).

Every number in the export is produced by the SAME trained model the local app
uses - nothing is re-fitted, approximated or invented here.

Usage:
    python export_static.py            # full build (~1 hour)
    python export_static.py --fast     # skip predict/robust (layout check)
    python export_static.py --only predict
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from core.config import BASE_DIR, MODEL_TARGETS, STATIC_DIR
from core.dataset import doe_levels, load_dataset

SITE_DIR = BASE_DIR / "docs"
API_DIR = SITE_DIR / "api"
REPO_URL = "https://github.com/MinJaeJun/chungbuk-ai-semiconductor"

# Pre-computed optimizer scenarios (MODE B snaps to the nearest of these).
N_TARGETS = 25
WEIGHTS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
ROBUST_SAMPLES = 1500
OPT_TOLERANCE = 0.05  # generous; the shim re-applies the user's tolerance
OPT_TOP_K = 40
CLOUD_CAP = 300


def dump(path: Path, obj: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    path.write_text(s, encoding="utf-8")
    return len(s.encode("utf-8"))


def human(n: int) -> str:
    return f"{n/1024/1024:.1f} MB" if n > 1024 * 1024 else f"{n/1024:.0f} KB"


# --------------------------------------------------------------------------- #
# frontend copy
# --------------------------------------------------------------------------- #
DEMO_CSS = """
/* ---- static demo banner (GitHub Pages build only) ---- */
.demo-bar {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 9px 24px; font-size: 12px; line-height: 1.55;
    background: linear-gradient(90deg, rgba(210,153,34,.16), rgba(163,113,247,.10));
    border-bottom: 1px solid rgba(210,153,34,.35); color: #f0d08a;
}
.demo-bar b { color: #fff; }
.demo-chip {
    font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: .6px;
    padding: 3px 9px; border-radius: 5px; background: var(--accent-yellow); color: #10131a;
    white-space: nowrap;
}
.demo-link {
    margin-left: auto; color: var(--accent-blue); text-decoration: none;
    font-family: var(--mono); font-size: 11.5px; white-space: nowrap;
}
.demo-link:hover { text-decoration: underline; }
"""


def copy_frontend() -> None:
    for sub in ("css", "js", "vendor"):
        dst = SITE_DIR / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(STATIC_DIR / sub, dst)

    css_path = SITE_DIR / "css" / "app.css"
    css_path.write_text(css_path.read_text(encoding="utf-8") + DEMO_CSS, encoding="utf-8")

    shim = SITE_DIR / "js" / "demo-shim.js"
    shim.write_text(shim.read_text(encoding="utf-8").replace("__REPO__", REPO_URL), encoding="utf-8")

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    # Pages serves from /<repo>/, so absolute /static/... paths would 404.
    html = html.replace('href="/static/', 'href="').replace('src="/static/', 'src="')
    # The shim must load after core.js defines API and before main.js boots.
    html = html.replace(
        '<script src="js/explorer.js"></script>',
        '<script src="js/demo-shim.js"></script>\n<script src="js/explorer.js"></script>',
    )
    html = html.replace(
        "<title>",
        '<meta name="description" content="TCAD 기반 이온주입·열처리 공정 AI Surrogate '
        '대시보드 정적 데모">\n<title>[DEMO] ',
    )
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")


# --------------------------------------------------------------------------- #
# payload builders
# --------------------------------------------------------------------------- #
def anonymize_variation(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip supplier-identifying strings from the variation reference.

    The statistics themselves (NU%, CV%, counts) are generic process-capability
    figures. What identifies the supplier is the recipe/equipment naming and the
    source filename, so only those are replaced. Every number is untouched,
    which keeps all claims in the UI true.
    """
    p = json.loads(json.dumps(payload))
    names = sorted({r["recipe"] for r in p.get("recipe_summary", [])})
    alias = {n: f"Recipe {chr(65 + i)}" for i, n in enumerate(names)}
    for r in p.get("recipe_summary", []):
        r["recipe"] = alias.get(r["recipe"], r["recipe"])
    for r in p.get("lot_to_lot", {}).get("per_recipe", []):
        r["recipe"] = alias.get(r["recipe"], r["recipe"])
    src = p.setdefault("source", {})
    src["file"] = "in-line thickness metrology (anonymized)"
    src["process"] = "Thin-film deposition, in-line thickness metrology"
    src["process_ko"] = "박막 증착 공정 · 인라인 두께 계측 (익명화)"
    p["anonymized"] = True
    return p


def build_simple(client, anonymize: bool = True) -> int:
    total = 0
    for endpoint, name in [
        ("/api/meta", "meta.json"),
        ("/api/dataset/summary", "dataset_summary.json"),
        ("/api/dataset/points", "dataset_points.json"),
        ("/api/xai/global", "xai_global.json"),
        ("/api/validation", "validation.json"),
        ("/api/variation", "variation.json"),
    ]:
        payload = client.get(endpoint).json()
        if name == "variation.json" and anonymize:
            payload = anonymize_variation(payload)
        if name == "dataset_points.json":
            # MODE A is recomputed in the browser from these rows and needs the
            # run_id to report which real TCAD run each recipe came from.
            df = load_dataset()
            payload["run_id"] = [int(v) for v in df["run_id"].to_numpy()]
        total += dump(API_DIR / name, payload)
    return total


def build_manifest(levels: dict[str, list[float]], targets: list[float]) -> int:
    return dump(
        API_DIR / "manifest.json",
        {
            "levels": levels,
            "optimize_ai": {"targets": targets, "weights": WEIGHTS,
                            "tolerance_used": OPT_TOLERANCE, "top_k": OPT_TOP_K},
            "robust": {"preset": "typical", "n_samples": ROBUST_SAMPLES},
            "note": "Pre-computed with the deployed surrogate; see export_static.py",
        },
    )


def build_whatif(client, levels) -> int:
    """One curve per (swept parameter, the other three levels)."""
    total = 0
    params = ["dose_cm2", "energy_keV", "anneal_temp_C", "anneal_time_sec"]
    for p in params:
        others = [k for k in params if k != p]
        bundle: dict[str, Any] = {}
        combos = [(a, b, c) for a in range(len(levels[others[0]]))
                  for b in range(len(levels[others[1]]))
                  for c in range(len(levels[others[2]]))]
        for (a, b, c) in combos:
            base = {
                others[0]: levels[others[0]][a],
                others[1]: levels[others[1]][b],
                others[2]: levels[others[2]][c],
                p: levels[p][0],
            }
            res = client.post("/api/whatif", json={**base, "parameter": p, "n_points": 41}).json()
            bundle[f"{a}_{b}_{c}"] = res
        n = dump(API_DIR / "whatif" / f"{p}.json", bundle)
        total += n
        print(f"    whatif/{p:<16s} {len(bundle):4d} curves  {human(n)}")
    return total


def build_optimize_ai(client, targets) -> int:
    total = 0
    for wi, w in enumerate(WEIGHTS):
        bundle: dict[str, Any] = {}
        for ti, t in enumerate(targets):
            r = client.post("/api/optimize", json={
                "target_xj_um": t, "tolerance_um": OPT_TOLERANCE, "rsh_mode": "minimize",
                "w_xj": w, "w_rsh": 1 - w, "mode": "ai", "top_k": OPT_TOP_K,
            }).json()
            r["cloud"] = {k: v[:CLOUD_CAP] for k, v in r["cloud"].items()}
            bundle[str(ti)] = r
        n = dump(API_DIR / "optimize_ai" / f"w{wi}.json", bundle)
        total += n
        print(f"    optimize_ai/w{wi} (w_xj={w:.1f})  {len(bundle)} targets  {human(n)}")
    return total


def build_grid(client, levels, kind: str) -> int:
    """predict / robust, bundled by (dose, energy) -> 25 anneal combinations."""
    total = 0
    nd, ne = len(levels["dose_cm2"]), len(levels["energy_keV"])
    nt, ns = len(levels["anneal_temp_C"]), len(levels["anneal_time_sec"])
    t0 = time.perf_counter()
    done = 0
    for d in range(nd):
        for e in range(ne):
            bundle: dict[str, Any] = {}
            for t in range(nt):
                for s in range(ns):
                    base = {
                        "dose_cm2": levels["dose_cm2"][d],
                        "energy_keV": levels["energy_keV"][e],
                        "anneal_temp_C": levels["anneal_temp_C"][t],
                        "anneal_time_sec": levels["anneal_time_sec"][s],
                    }
                    if kind == "predict":
                        res = client.post("/api/predict", json={**base, "explain": True}).json()
                    else:
                        # The spec only affects yield/Cpk, which the shim
                        # recomputes from the stored histogram, so any spec
                        # works here as the seed.
                        res = client.post("/api/robust", json={
                            **base, "target_xj_um": 0.25, "tolerance_um": 0.004,
                            "rsh_max": 100.0, "preset": "typical", "n_samples": ROBUST_SAMPLES,
                        }).json()
                    bundle[f"{t}_{s}"] = res
                    done += 1
            total += dump(API_DIR / kind / f"d{d}e{e}.json", bundle)
        el = time.perf_counter() - t0
        pct = done / (nd * ne * nt * ns) * 100
        eta = el / max(done, 1) * (nd * ne * nt * ns - done)
        print(f"    {kind}: {done:4d}/{nd*ne*nt*ns}  {pct:5.1f}%  elapsed {el/60:.1f}m  eta {eta/60:.1f}m", flush=True)
    return total


def verify_mode_a(client, levels) -> None:
    """MODE A is recomputed in JS; make sure the inputs it needs are exported
    and that the backend answer is reproducible from the shipped rows alone."""
    import numpy as np

    df = load_dataset()
    tb = {c: (float(df[c].min()), float(df[c].max())) for c in ("xj_final_um", "rsh_final_ohm_sq")}
    for target, w in [(0.25, 0.6), (0.30, 1.0), (0.20, 0.0)]:
        api = client.post("/api/optimize", json={
            "target_xj_um": target, "tolerance_um": 0.01, "rsh_mode": "minimize",
            "w_xj": w, "w_rsh": 1 - w, "mode": "doe", "top_k": 5,
        }).json()
        xj = df["xj_final_um"].to_numpy()
        rsh = df["rsh_final_ohm_sq"].to_numpy()
        err = np.abs(xj - target)
        span = tb["xj_final_um"][1] - tb["xj_final_um"][0]
        lo, hi = tb["rsh_final_ohm_sq"]
        score = (w * err / span + (1 - w) * np.clip((rsh - lo) / (hi - lo), 0, None)) / (w + (1 - w) or 1)
        feas = err <= 0.01
        order = np.lexsort((score, ~feas))[:5]
        mine = [int(df["run_id"].to_numpy()[i]) for i in order]
        theirs = [r["run_id"] for r in api["recipes"]]
        assert mine == theirs, f"MODE A mismatch at target={target}, w={w}: {mine} vs {theirs}"
    print("    MODE A client-side formula verified against backend (3 scenarios)")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip predict/robust")
    ap.add_argument("--only", choices=["frontend", "simple", "whatif", "optimize", "robust", "predict"])
    # Anonymised by default: the built site is the artefact most likely to be
    # copied or published, so it must never carry supplier-identifying strings.
    ap.add_argument("--keep-names", action="store_true",
                    help="keep original recipe/equipment names in the exported "
                         "variation reference (local inspection only)")
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    import app as app_module

    client = TestClient(app_module.app)
    levels = doe_levels()
    df = load_dataset()
    lo, hi = float(df["xj_final_um"].min()), float(df["xj_final_um"].max())
    targets = [round(lo + (hi - lo) * i / (N_TARGETS - 1), 5) for i in range(N_TARGETS)]

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    stages = [args.only] if args.only else (
        ["frontend", "simple", "whatif", "optimize"] if args.fast
        else ["frontend", "simple", "whatif", "optimize", "robust", "predict"]
    )

    print("=" * 74)
    print(" Static demo export ->", SITE_DIR)
    print("=" * 74)
    total = 0
    t_all = time.perf_counter()

    if "frontend" in stages:
        print("\n[1] frontend")
        copy_frontend()
        print("    index.html / css / js / vendor copied, shim injected")

    if "simple" in stages:
        print("\n[2] static endpoints")
        n = build_simple(client, anonymize=not args.keep_names)
        n += build_manifest(levels, targets)
        total += n
        print(f"    {human(n)}" + ("" if args.keep_names else "  [anonymized]"))
        verify_mode_a(client, levels)

    if "whatif" in stages:
        print("\n[3] what-if sweeps")
        total += build_whatif(client, levels)

    if "optimize" in stages:
        print(f"\n[4] optimizer MODE B ({N_TARGETS} targets x {len(WEIGHTS)} weights)")
        total += build_optimize_ai(client, targets)

    if "robust" in stages:
        print(f"\n[5] robustness ({ROBUST_SAMPLES} samples/point, preset=typical)")
        total += build_grid(client, levels, "robust")

    if "predict" in stages:
        print("\n[6] predictions (with exact SHAP)")
        total += build_grid(client, levels, "predict")

    size = sum(f.stat().st_size for f in SITE_DIR.rglob("*") if f.is_file())
    print("\n" + "=" * 74)
    print(f" wrote {human(total)} of API payload   |   site total {human(size)}")
    print(f" elapsed {(time.perf_counter()-t_all)/60:.1f} min")
    print(f" open locally:  python -m http.server -d docs 8080")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

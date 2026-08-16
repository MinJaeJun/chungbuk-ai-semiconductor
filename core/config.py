"""Project-wide paths and process-domain constants.

AI Semiconductor Process Optimizer
TCAD-based Ion Implantation & Annealing Process Prediction / Optimization Platform
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

CSV_NAME = "implant_anneal_1000.csv"
CSV_PATH = DATA_DIR / CSV_NAME

ARTIFACT_PATH = MODEL_DIR / "surrogate_bundle.joblib"
REPORT_PATH = OUTPUT_DIR / "training_report.json"

# ---------------------------------------------------------------------------
# Leakage guard: ONLY these four physically controllable process parameters may
# ever enter the model as inputs.  run_id and every measured/simulated output
# column are forbidden as features.
# ---------------------------------------------------------------------------
PROCESS_INPUTS: list[str] = [
    "dose_cm2",
    "energy_keV",
    "anneal_temp_C",
    "anneal_time_sec",
]

FORBIDDEN_AS_FEATURE: list[str] = [
    "run_id",
    "xj_implant_um",
    "xj_final_um",
    "delta_xj_um",
    "rsh_final_ohm_sq",
]

# Model feature matrix column order (dose is log10-transformed: it spans
# 5e14 .. 2.5e15 and Rsh scales ~1/dose, so log10 linearises the relationship).
FEATURE_COLUMNS: list[str] = [
    "log10_dose",
    "energy_keV",
    "anneal_temp_C",
    "anneal_time_sec",
]

# Mapping feature-matrix column -> human readable process variable
FEATURE_LABELS: dict[str, str] = {
    "log10_dose": "Implant Dose",
    "energy_keV": "Implant Energy",
    "anneal_temp_C": "Anneal Temperature",
    "anneal_time_sec": "Anneal Time",
}

FEATURE_LABELS_KO: dict[str, str] = {
    "log10_dose": "이온주입 Dose",
    "energy_keV": "이온주입 Energy",
    "anneal_temp_C": "열처리 온도",
    "anneal_time_sec": "열처리 시간",
}

# Directly modelled targets. delta_xj_um is NOT modelled directly - it is
# derived as (predicted xj_final - predicted xj_implant) because the dataset
# satisfies delta = final - implant exactly.
MODEL_TARGETS: list[str] = [
    "xj_implant_um",
    "xj_final_um",
    "rsh_final_ohm_sq",
]

TARGET_META: dict[str, dict[str, str]] = {
    "xj_implant_um": {
        "label": "Xj Implant",
        "label_ko": "이온주입 직후 접합깊이",
        "unit": "um",
        "role": "auxiliary",
    },
    "xj_final_um": {
        "label": "Xj Final",
        "label_ko": "열처리 후 최종 접합깊이",
        "unit": "um",
        "role": "main",
    },
    "rsh_final_ohm_sq": {
        "label": "Rsh Final",
        "label_ko": "면저항",
        "unit": "ohm/sq",
        "role": "main",
    },
    "delta_xj_um": {
        "label": "Delta Xj",
        "label_ko": "열처리 확산량",
        "unit": "um",
        "role": "derived",
    },
}

INPUT_META: dict[str, dict[str, str]] = {
    "dose_cm2": {"label": "Implant Dose", "label_ko": "이온주입 도즈", "unit": "cm^-2"},
    "energy_keV": {"label": "Implant Energy", "label_ko": "이온주입 에너지", "unit": "keV"},
    "anneal_temp_C": {"label": "Anneal Temperature", "label_ko": "열처리 온도", "unit": "degC"},
    "anneal_time_sec": {"label": "Anneal Time", "label_ko": "열처리 시간", "unit": "sec"},
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

DISCLAIMER_EN = (
    "This AI model is trained on TCAD-generated simulation data. "
    "Actual semiconductor manufacturing deployment requires additional Fab data "
    "and physical validation."
)
DISCLAIMER_KO = (
    "본 AI 모델은 TCAD 시뮬레이션으로 생성된 DOE 데이터로 학습되었습니다. "
    "실제 반도체 양산 라인 적용을 위해서는 추가적인 Fab 계측 데이터 확보와 "
    "물리적 검증(Split 실험)이 반드시 필요합니다."
)


def ensure_dirs() -> None:
    for path in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)

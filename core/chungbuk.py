"""Regional fit analysis for Chungcheongbuk-do's semiconductor industry.

Source
------
`data/chungbuk_semiconductor_firms_20250624.csv` - the Chungcheongbuk-do
semiconductor company register published as public open data (snapshot
2025-06-24), 588 firms with head-office address, main product and KSIC
industry label.

Why it is here
--------------
The surrogate itself is demonstrated on ion implantation, a front-end process.
Chungbuk's semiconductor base is not front-end fabs: it is dominated by
equipment, parts and packaging suppliers, and the register lets that be stated
as a measured fact instead of an assertion.  Every figure this module reports
is computed from the CSV at run time.

Nothing in this module feeds the surrogate.  It is descriptive context for the
deployment case, kept strictly separate from anything the model learns.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR, ensure_dirs

FIRMS_CSV = DATA_DIR / "chungbuk_semiconductor_firms_20250624.csv"
REGIONAL_REPORT = OUTPUT_DIR / "chungbuk_regional_report.json"

CITIES = (
    "청주",
    "충주",
    "제천",
    "보은",
    "옥천",
    "영동",
    "증평",
    "진천",
    "괴산",
    "음성",
    "단양",
)

# KSIC labels that place a firm directly in the semiconductor value chain.
SEMICONDUCTOR_PATTERN = (
    r"반도체|전자집적회로|다이오드|트랜지스터|웨이퍼|표시장치|발광 ?다이오드"
)

# Within that set, the sub-segments Chungbuk actually concentrates in.
SEGMENT_PATTERNS: dict[str, str] = {
    "장비·부품 (Equipment & Parts)": r"반도체 ?제조용 ?(?:기계|장치)|반도체ㅤ?LCD장비|반도체장비",
    "소자·집적회로 (Device & IC)": r"집적회로|다이오드|트랜지스터|반도체소자|반도체 ?소자|반도체 ?제조업$",
    "디스플레이 (Display)": r"표시장치",
}

# Firms whose KSIC marks them as software houses - the local integration
# partners any AI process tool would actually be deployed through.
SOFTWARE_PATTERN = r"소프트웨어"

_CACHE: dict[str, Any] = {}


def load_firms(force: bool = False) -> pd.DataFrame:
    """Read the regional company register (cached)."""
    if force or "firms" not in _CACHE:
        if not FIRMS_CSV.exists():
            raise FileNotFoundError(
                f"Chungbuk company register not found: {FIRMS_CSV}"
            )
        frame = pd.read_csv(FIRMS_CSV, encoding="utf-8")
        frame["시군"] = frame["본사주소"].map(_city_of)
        frame["업종명"] = frame["업종명"].fillna("")
        _CACHE["firms"] = frame
    return _CACHE["firms"]


def _city_of(address: Any) -> str:
    if not isinstance(address, str):
        return "미상"
    found = re.search("|".join(CITIES), address)
    return found.group(0) if found else "기타"


def _normalise(label: str) -> str:
    """KSIC labels in the register vary by spacing ('그외'/'그 외')."""
    return re.sub(r"\s+", " ", label).strip()


def regional_profile() -> dict[str, Any]:
    """Measured composition of the Chungbuk semiconductor base."""
    frame = load_firms()
    total = int(len(frame))

    by_city = frame["시군"].value_counts()
    cities = [
        {"시군": city, "firms": int(count), "share_pct": round(100.0 * count / total, 1)}
        for city, count in by_city.items()
    ]

    core = frame[frame["업종명"].str.contains(SEMICONDUCTOR_PATTERN, regex=True)]
    core_total = int(len(core))

    segments = []
    assigned = pd.Series(False, index=core.index)
    for name, pattern in SEGMENT_PATTERNS.items():
        hit = core["업종명"].str.contains(pattern, regex=True) & ~assigned
        assigned |= hit
        segments.append(
            {
                "segment": name,
                "firms": int(hit.sum()),
                "share_of_core_pct": round(100.0 * int(hit.sum()) / core_total, 1)
                if core_total
                else 0.0,
            }
        )
    other = core_total - int(assigned.sum())
    if other:
        segments.append(
            {
                "segment": "기타 (Other)",
                "firms": other,
                "share_of_core_pct": round(100.0 * other / core_total, 1),
            }
        )

    listed = frame["기업분류"].fillna("미분류").value_counts()
    software = int(frame["업종명"].str.contains(SOFTWARE_PATTERN, regex=True).sum())

    top_industries = [
        {"업종명": _normalise(str(label)), "firms": int(count)}
        for label, count in frame["업종명"].map(_normalise).value_counts().head(10).items()
    ]

    return {
        "source": FIRMS_CSV.name,
        "snapshot": "2025-06-24",
        "total_firms": total,
        "by_city": cities,
        "cheongju_share_pct": next(
            (c["share_pct"] for c in cities if c["시군"] == "청주"), None
        ),
        "core_semiconductor_firms": core_total,
        "core_segments": segments,
        "company_class": {
            str(k): int(v) for k, v in listed.items()
        },
        "listed_firms": int(listed.get("IPO", 0)),
        "listed_share_pct": round(100.0 * int(listed.get("IPO", 0)) / total, 2),
        "software_firms": software,
    }


def deployment_case() -> dict[str, Any]:
    """Where this tool fits the measured regional profile, stated conservatively."""
    profile = regional_profile()
    equipment = next(
        (
            s
            for s in profile["core_segments"]
            if s["segment"].startswith("장비·부품")
        ),
        None,
    )
    return {
        "profile": profile,
        "observations": [
            (
                f"충북 반도체 관련 기업 {profile['total_firms']}개 중 "
                f"{profile['cheongju_share_pct']}%가 청주에 집중되어 있어, "
                "단일 거점 대상 실증이 가능한 밀집도입니다."
            ),
            (
                f"반도체 직결 업종 {profile['core_semiconductor_firms']}개 가운데 "
                f"{equipment['firms']}개({equipment['share_of_core_pct']}%)가 장비·부품 "
                "제조업입니다. 충북의 강점은 소자 양산이 아니라 장비·부품·후공정입니다."
            ),
            (
                f"상장사는 {profile['listed_firms']}개({profile['listed_share_pct']}%)에 "
                "불과합니다. 상용 TCAD 라이선스를 상시 보유하기 어려운 중소기업이 "
                "절대다수라는 뜻입니다."
            ),
            (
                f"지역 내 소프트웨어 개발·공급 기업이 {profile['software_firms']}개 있어, "
                "도입 시 외부 SI에 의존하지 않는 지역 파트너 풀이 존재합니다."
            ),
        ],
        "fit": (
            "본 시스템의 대상은 TCAD 라이선스를 상시 운용하기 어려운 장비·부품·후공정 "
            "중소기업입니다. 이미 보유한 DOE 결과만으로 조건 탐색과 산포 견딤 평가를 "
            "수행하게 하는 것이 목적이며, TCAD를 대체하지 않습니다."
        ),
        "caveat": (
            "이 분석은 공개된 기업 등록 정보의 업종 분류만을 집계한 것입니다. "
            "개별 기업의 TCAD 보유 여부, 공정 데이터 보유 여부, 도입 의향은 "
            "확인된 바 없으며 수요를 추정하지 않았습니다."
        ),
    }


def save_report(path=REGIONAL_REPORT) -> Any:
    ensure_dirs()
    payload = deployment_case()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

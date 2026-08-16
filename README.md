# AI Semiconductor Process Optimizer

**TCAD 기반 Ion Implantation & Annealing 공정 예측·최적화 플랫폼**
_AI 기반 반도체 이온주입·열처리 공정 최적화 플랫폼_

> From Simulation Data to Optimized Recipe
> TCAD × Machine Learning × Process Optimization

2026년 지역주도 디지털혁신지원사업 제13회 전국 ICT융합 공모전 · **AI 기반 산업 혁신** 분야 제출용 MVP

> **포지셔닝**
> 본 시스템은 **TCAD 기반 공정 의사결정 지원용 AI Surrogate Model**입니다.
> 실제 Fab 공정이나 TCAD 시뮬레이션을 대체하는 도구가 아니라,
> **공정 조건 후보를 빠르게 탐색하기 위한 AI 지원 시스템**입니다.

> **데이터 확보 현황 (중요)**
> 충북 반도체 기업 대상 데이터 제공 요청에 대한 회신이 없어, **신규 기업 데이터 없이**
> 현재 보유 데이터만으로 프로젝트를 완결했습니다. 본 저장소는 외부 데이터 유입 없이
> 그대로 학습·실행·검증이 가능하며, 아래 모든 수치는 동봉된 두 데이터 파일에서 계산됩니다.

---

## 팀원용 빠른 시작 (2분)

**A. 화면만 보고 싶다 — Python 패키지 설치 불필요**

```bash
git clone https://github.com/MinJaeJun/chungbuk-ai-semiconductor.git
cd chungbuk-ai-semiconductor
python -m http.server -d docs 8080
# 브라우저에서 http://127.0.0.1:8080
```

`docs/`는 학습된 모델의 추론 결과를 **사전 계산해 넣은 정적 데모**입니다.
Data Explorer · What-If · Optimizer(MODE A) · XAI · Robustness · Model Validation이
그대로 동작하고, 공정 조건은 TCAD DOE 격자점 1,000개로 스냅됩니다.

**B. 전체 기능을 쓰고 싶다 — 임의 조건 예측 · MODE B 자유 탐색 · 산포 재평가**

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn app:app --port 8000
# 브라우저에서 http://127.0.0.1:8000
```

학습된 모델(`models/surrogate_bundle.joblib`)이 저장소에 포함되어 있으므로
**`train_model.py`를 다시 돌릴 필요가 없습니다**(재학습 시 약 19분).

> **저장소에 없는 파일 하나**
> `data/fab_thickness_profile_17000.csv`(기업 제공 원본 계측 17,000행)는
> **의도적으로 커밋하지 않습니다.** 대신 파생 집계인
> `outputs/variation_reference.json`(비율·개수만 포함, 웨이퍼별 측정값 없음)이
> 버전 관리되며, 이것만으로 Robustness 기능이 모두 동작합니다.
> 원본을 다시 계산하려면 CSV를 `data/`에 넣고
> `python -c "from core import variation; variation.analyze(force=True)"` 를 실행하십시오.

---

## 1. 프로젝트 소개

반도체 소자의 Source/Drain, Well 형성 공정에서 엔지니어는
**Dose · Implant Energy · Annealing Temperature · Annealing Time** 네 가지 조건을
반복적으로 바꿔가며 **Junction Depth(Xj)** 와 **Sheet Resistance(Rsh)** 를 맞춰야 합니다.

이 프로젝트는 TCAD로 생성된 1,000건의 Full Factorial DOE를 머신러닝이 학습하여
**TCAD의 Surrogate Model** 역할을 하게 하고, 여기에 **실측 라인에서 측정한 공정 산포**를
결합하여 "목표에 맞는 조건"이 아니라 **"산포를 견디는 조건"** 을 추천합니다.

```
TCAD Simulation → Structured DOE Data → AI Surrogate Model → Fast Prediction
   → Explainable AI → Process Window Exploration → Multi-objective Optimization
   → 실측 산포 기반 Robustness 평가 → Recommended Process Recipe
   → Engineer Decision Support
```

---

## 2. 문제 정의

| 항목 | 현재 방식 | 본 시스템 |
|---|---|---|
| 조건 1건 평가 | TCAD 시뮬레이션 (수 분~수십 분) | 학습된 surrogate 추론 (**수 ms**) |
| 조건 후보 탐색 | 엔지니어가 수동으로 나열 | **28,000+ 조건** 자동 탐색 후 Top-N |
| 변수 영향도 | 경험 + 개별 sweep 반복 | Permutation Importance + **정확 SHAP** |
| Trade-off 판단 | 정성적 | **Pareto Frontier** 시각화 |
| **산포 고려** | **별도 split 실험 필요** | **실측 산포 기반 Monte-Carlo 수율·Cpk** |

---

## 3. 데이터셋 (2종, 모두 동봉)

### 3-1. `data/implant_anneal_1000.csv` — AI 학습용 Ground Truth

TCAD 시뮬레이션 DOE. **이 파일이 예측 모델의 유일한 학습 데이터입니다.**

| 항목 | 값 |
|---|---|
| 총 행 수 | **1,000 runs** / 9 컬럼 |
| 결측치 · 중복 | **0 · 0** |
| 설계 | **5 × 8 × 5 × 5 = 1000 Full Factorial (완전 균형)** |

**Process Input (모델 입력 · 4개)**

| 변수 | 단위 | 수준 |
|---|---|---|
| `dose_cm2` | cm⁻² | 5.0e14 · 1.0e15 · 1.5e15 · 2.0e15 · 2.5e15 |
| `energy_keV` | keV | 10 · 12.5 · 15 · 17.5 · 20 · 30 · 40 · 50 |
| `anneal_temp_C` | °C | 900 · 950 · 1000 · 1050 · 1100 |
| `anneal_time_sec` | sec | 5 · 15 · 25 · 35 · 45 |

**Output**

| 변수 | 단위 | 관측 범위 | 역할 |
|---|---|---|---|
| `xj_implant_um` | um | 0.12223 ~ 0.45499 | AI Target (Auxiliary) |
| `xj_final_um` | um | 0.12223 ~ 0.45717 | **AI Target (Main)** |
| `rsh_final_ohm_sq` | ohm/sq | 29.841 ~ 387.09 | **AI Target (Main)** |
| `delta_xj_um` | um | -3.00e-6 ~ 0.04182 | **파생값(학습하지 않음)** |

**데이터에서 직접 검증한 사실**

* `delta_xj_um = xj_final_um - xj_implant_um` 항등식 성립 (최대 잔차 **5.13e-17**)
  → ΔXj는 독립 target으로 학습하지 않고 예측된 두 접합깊이의 차이로 계산
* `xj_implant_um`은 (dose, energy) 40개 조합에 값이 정확히 1개씩만 존재
  → 이온주입 직후 접합깊이는 **anneal 조건과 무관**하며, XAI에서도 재현됨
  (anneal_temp / anneal_time의 SHAP 기여도 = **정확히 0**)

**Data Leakage 방지 정책**

```
features = [dose_cm2, energy_keV, anneal_temp_C, anneal_time_sec]
excluded = [run_id, xj_implant_um, xj_final_um, delta_xj_um, rsh_final_ohm_sq]
```

시뮬레이션 이전에 지정 가능한 제어 변수만 입력으로 사용합니다. `dose_cm2`는 log10 변환합니다.
정책은 `core/config.py`에 상수로 고정되고 `tests/test_pipeline.py`가 강제합니다.

### 3-2. `data/fab_thickness_profile_17000.csv` — 실측 공정 산포 레퍼런스

**실제 양산 라인의 인라인 계측 데이터입니다. 예측 모델 학습에는 사용하지 않습니다.**

| 항목 | 값 |
|---|---|
| 공정 | PECVD SiON 증착 · 인라인 두께 계측 |
| 규모 | **17,000 계측 · 40 lot · 1,000 wafer · 6 recipe** |
| 웨이퍼당 측정 | 13점 맵 (+ edge item) |
| 기간 | 2024-11-06 ~ 2024-11-08 |
| 단위 | Å (1589.85 ~ 2471.36) |

**측정된 실제 공정 산포 (`core/variation.py`가 CSV에서 직접 계산)**

| 지표 | 정의 | 실측값 |
|---|---|---|
| Within-wafer NU% | (max−min)/(2·mean), 13점 맵 | mean **7.298** · median 8.860 · p90 9.692 |
| Wafer-to-wafer CV% | lot 내 웨이퍼 평균의 std/mean | mean **0.653** · median 0.487 · p90 **1.378** |
| Lot-to-lot CV% | recipe 내 lot 평균의 std/mean | mean **1.256** |
| Pooled total CV% | 전체 1,000 웨이퍼 평균의 std/mean | **2.036** |

> **범위 주의 (반드시 함께 언급)**
> 이 산포는 **이온주입이 아닌 PECVD SiON 증착 공정**의 실측값입니다.
> 공정 입력 tolerance의 **크기(order of magnitude)를 정할 때 참조 기준**으로만 사용하며,
> 이온주입 공정 모델로 사용하지 않습니다. UI·API·코드 주석 모두에 이 경고가 포함되어 있고,
> 모든 tolerance는 사용자가 직접 덮어쓸 수 있습니다.

---

## 4. AI Model 구조

### 4.1 후보 모델 (임의 선택이 아닌 실측 비교)

Linear Regression · Polynomial(deg2/deg3)+RidgeCV · Random Forest · Extra Trees ·
Gradient Boosting · Hist Gradient Boosting · **Gaussian Process (Matérn 5/2)** · MLP(64-64-32)
+ Rsh에는 log10 target 변환 변형 7종 추가 → **Xj 9종 / Rsh 16종** 비교

XGBoost / LightGBM은 설치 시 자동으로 후보에 추가되지만 **없어도 완전히 동작합니다.**

### 4.2 검증 프로토콜 (2중 교차검증)

Full Factorial 설계 때문에 (dose, energy) 조합 하나가 anneal 격자를 따라 **25번 반복**됩니다.
특히 `xj_implant_um`은 (dose, energy)만의 결정론적 함수라서
**단순 무작위 분할에서는 트리 모델이 매핑을 암기하여 R² = 1.000000이 나옵니다.**
이를 성능이라고 주장하는 것은 정직하지 않으므로 두 프로토콜을 모두 측정·표시합니다.

| 프로토콜 | 내용 | 의미 |
|---|---|---|
| **Random Hold-out** | 80/20 무작위 분할 (seed 42) + 5-fold KFold | 격자 내부 보간 성능 |
| **Group CV** | **GroupKFold on 40개 (dose, energy) 조건** | **학습 중 본 적 없는 이온주입 조건** 일반화 |

### 4.3 모델 선택 기준

* `rsh_final_ohm_sq` → **Group CV R² 최대**
* `xj_implant_um` / `xj_final_um` → **쌍(pair)으로 선택.**
  Group CV R²가 각 target 최고값의 0.005 이내인 후보들(5 × 6 = 30쌍) 중
  **파생 ΔXj의 hold-out RMSE가 최소가 되는 조합**을 선택합니다.
  ΔXj는 학습하지 않는 값이므로 그 정확도는 모델 쌍의 실측 가능한 고유 성질입니다.

선정 모델은 마지막에 **전체 1,000 run으로 재학습**되어 배포됩니다.

### 4.4 최종 선정 모델 & 실측 성능

> 아래 모든 수치는 `python train_model.py` 실행 시 실제 데이터에서 계산되어
> `outputs/training_report.json`에 저장된 값입니다. **하드코딩된 성능 수치는 없습니다.**
> (Python 3.12.10 / scikit-learn 1.9.0 / train 800 · test 200 / 총 학습 1,142초)

| Target | 선정 모델 | 검증 | **R²** | **MAE** | **RMSE** |
|---|---|---|---|---|---|
| **xj_final_um** [um] | Gaussian Process (Matérn 5/2) | Hold-out Test (20%) | **0.99998842** | **1.0579e-4** | **3.3424e-4** |
| | | Random CV (5-fold) | 0.999984 | 1.5731e-4 | 4.1055e-4 |
| | | Group CV (미지 조건) | 0.993232 | 5.4588e-3 | 8.4285e-3 |
| **rsh_final_ohm_sq** [ohm/sq] | Gaussian Process (Matérn 5/2) | Hold-out Test (20%) | **0.99996267** | **0.13223** | **0.29147** |
| | | Random CV (5-fold) | 0.999882 | 0.22809 | 0.56819 |
| | | Group CV (미지 조건) | 0.998456 | 0.97169 | 2.05459 |
| **xj_implant_um** [um] | Extra Trees | Hold-out Test (20%) | **0.99999925** | **3.3203e-5** | **8.5333e-5** |
| | | Random CV (5-fold) | 0.999997 | 5.2484e-5 | 1.7217e-4 |
| | | Group CV (미지 조건) | 0.994688 | 5.9796e-3 | 7.5019e-3 |
| **delta_xj_um** [um] *(파생, 미학습)* | GP − Extra Trees | Hold-out Test (20%) | **0.95999** | **1.0658e-4** | **3.3139e-4** |

* Test MAPE: Xj Final **0.056 %**, Rsh **0.129 %**, Xj Implant **0.013 %**
* 배포 모델의 TCAD 재현: Xj 최대 절대오차 **8.8e-8 um**, Rsh **1.0e-5 ohm/sq**
* 물리적으로 기대되는 단조성이 보간 구간에서도 유지됨:
  Xj Final ↑ (energy ↑), Rsh ↓ (dose ↑), Rsh ↓ (anneal temp ↑)

전체 비교표는 앱 **Model Validation** 탭과 `outputs/leaderboard_*.csv`에 있습니다.

---

## 5. Prediction 방법

1. Dose / Energy / Anneal Temp / Anneal Time 입력 (슬라이더 + DOE 격자 칩)
2. `POST /api/predict` → **Xj Implant · Xj Final · ΔXj · Rsh** 예측
3. 각 값에 hold-out RMSE 기반 불확실성 범위(±) 표기
4. 입력 위치를 3단계로 판정

| 단계 | 조건 | 표시 |
|---|---|---|
| `validated_doe_point` | 4개 값이 모두 DOE 격자점과 일치 | ✔ 검증된 영역 |
| `interpolation` | 학습 범위 내부, 격자 사이 | ℹ AI 보간 · TCAD 확인 권장 |
| `extrapolation` | 하나라도 학습 범위 밖 | ⚠ **EXTRAPOLATION WARNING** |

5. **ΔXj 분해능 처리** — `|ΔXj| < 3.31e-4 um`(파생 ΔXj hold-out RMSE)이면
   **"분해능 미만, 값 자체를 해석하지 말 것"** 경고를 표시합니다.
6. 가장 가까운 실제 TCAD run과 나란히 비교

---

## 6. Optimization 방법

### MODE A · Validated DOE Search
TCAD가 실제 시뮬레이션한 **1,000개 조건의 측정값**으로 순위를 매깁니다.
예측이 개입하지 않아 추천 recipe가 모두 실제 run으로 뒷받침됩니다.

### MODE B · AI Interpolation Search
Coarse grid(11×13×13×13) → 상위 20개 주변 refinement(5⁴) 2단계로 **약 28,000 조건** 평가.
결과에는 항상 다음 경고가 붙습니다.

> **AI interpolation result — TCAD/Fab verification required**

### 다목적 스코어

```
score = ( w_xj · |Xj − Xj_target| / range(Xj)  +  w_rsh · (Rsh − min) / range(Rsh) ) / (w_xj + w_rsh)
```

* 제약: `|Xj − Xj_target| ≤ tolerance`, 선택적 `Rsh ≤ 상한`
* 변수 1개 DOE 수준 **고정(lock)** 가능
* **Pareto Frontier**로 비지배 후보 시각화
* Top-N은 중복 제거 + (MODE B) 정규화 DOE 공간 최소 이격 6% 로 **서로 다른 공정 윈도우** 보장

---

## 7. Process Robustness — 실측 산포 기반 수율 평가

> **이 프로젝트에서 가장 중요한 차별점입니다.**

TCAD DOE는 결정론적이라 한 조건에 답이 하나뿐이고 surrogate는 이를 1e-8 수준으로 재현합니다.
그러나 실제 라인은 모든 setpoint에 산포가 실립니다.
**산포를 고려하지 않은 "최적 조건"은 목표값에 정확히 맞으면서도 웨이퍼 대부분을 spec 밖으로 흘려보낼 수 있습니다.**

### 방법

1. 실측 CSV(17,000행)에서 공정 산포 크기를 측정 (§3-2)
2. 그 크기를 **참조 기준**으로 삼아 4개 공정 입력에 상대 1σ tolerance를 부여
3. 학습 범위 내부로 clip한 Gaussian 샘플 2,500개를 surrogate에 통과
4. **Spec 수율 · Cpk · 분포 · 산포 기여도(variance decomposition)** 산출

### Tolerance Preset (실측 기준 앵커)

| Preset | 앵커 | dose | energy | temp | time |
|---|---|---|---|---|---|
| Tight | 실측 W2W CV mean (0.65%) | 0.65% | 0.30% | 0.25% | 1.0% |
| **Typical** | 실측 W2W CV p90 (1.38%) | 1.40% | 0.60% | 0.50% | 2.0% |
| Conservative | 실측 pooled total CV (2.04%) | 2.04% | 1.00% | 0.80% | 4.0% |

모든 값은 UI에서 개별 조정 가능하며, 실제 이온주입기 tolerance로 교체해야 합니다.

### 실행 결과 예 (Dose 1.5e15 · 20 keV · 1000 °C · 25 s, Typical, Xj 0.2452 ± 0.004)

* Joint Spec 수율 **99.64 %**, Cpk(Xj) **1.05**
* **Xj 산포의 97.2 %가 Implant Energy 산포**에서 발생
* **Rsh 산포의 60.8 %가 Anneal Temperature 산포**에서 발생
  → 수율을 올리려면 어떤 장비의 제어를 먼저 개선해야 하는지가 숫자로 나옵니다.

### 정직성 장치 두 가지

1. **경계 clipping 경고** — setpoint가 DOE 경계(예: dose 2.5e15)에 있으면 산포 샘플의 절반이
   학습 범위 밖으로 나가 clip됩니다. 이 경우 산포가 실제보다 좁게 평가되어 **수율이 과대평가**되므로,
   clip된 비율을 측정해 1% 초과 시 경고를 띄웁니다. (예: dose 경계 조건에서 49.9% clip 감지)
2. **두 가지 불확실성 분리 표시**
   * ① **모델 불확실성** — GP posterior std (surrogate가 자기 답을 얼마나 확신하지 못하는가)
   * ② **공정 산포** — 장비 산포가 만드는 결과의 흩어짐
   * 실측 기준 조건에서 **② / ① ≈ 1,176배** → **지금 병목은 모델 정확도가 아니라 공정 제어**라는 결론.
     모델을 더 키우는 것보다 장비 tolerance를 줄이는 편이 수율에 직접 기여합니다.

### Optimizer 연동 · Nominal vs Robust

Optimizer 탭에서 "+ 산포 재평가"를 켜면 Top-N을 Monte-Carlo로 재평가해 **수율 기준 순위**를 함께 보여줍니다.
Conservative tolerance · Xj 0.25 ± 0.004 조건 실측 결과, **8개 중 5개의 순위가 바뀌었습니다.**

| Nominal 순위 | Robust 순위 | Spec 수율 |
|---|---|---|
| 1 | 1 | 94.62 % |
| **7** | **4** (▲3) | 92.75 % |
| **3** | **6** (▼3) | 90.38 % |
| 5 | 7 (▼2) | 81.25 % |
| 8 | 8 | 70.38 % |

**목표값에 가장 정확히 맞는 조건이 산포에 가장 강건한 조건은 아닙니다.**

---

## 8. XAI 설명

| 수준 | 방법 | 구현 |
|---|---|---|
| **Global** | Permutation Feature Importance | 30 repeats, scoring=R², **hold-out test set** |
| **Global** | Mean \|SHAP\| | DOE 샘플에 대한 정확 Shapley value 평균 절대값 |
| **Local** | **정확(Exact) Interventional Shapley Value** | 입력 4개 → **2⁴ = 16 coalition 전수 평가** |

`shap` 라이브러리 없이 numpy만으로 **근사가 아닌 정확한 SHAP**을 계산합니다.
efficiency 공리가 수치적으로 성립합니다 (실측 잔차 ≈ 1e-18, 테스트로 강제).

```
Σφ_j = f(x) − E[f(X)]
```

**실측 Global SHAP (mean |φ|)**

| Target | 1위 | 2위 | 3위 | 4위 |
|---|---|---|---|---|
| `rsh_final_ohm_sq` | Dose **35.41** | Anneal Temp **14.54** | Energy **11.36** | Anneal Time **4.68** |
| `xj_final_um` | Energy **0.0806** | Dose **0.0042** | Anneal Temp **9.8e-4** | Anneal Time **5.1e-4** |
| `xj_implant_um` | Energy **0.0810** | Dose **0.0042** | Anneal Temp **0** | Anneal Time **0** |

`xj_implant_um`의 anneal 기여도가 **정확히 0**인 것은
"이온주입 직후 접합깊이는 열처리와 무관"이라는 데이터 구조를 모델이 그대로 학습했다는 직접 증거입니다.

---

## 9. AI Process Insight (3단계 근거 구분)

| 레벨 | 근거 |
|---|---|
| `DATA OBSERVATION` | CSV에서 직접 계산 (균형 설계 주효과 + Spearman 순위상관) |
| `MODEL INTERPRETATION` | 학습된 surrogate + XAI 결과 |
| `ENGINEERING VERIFICATION REQUIRED` | 실측 오차 + 외삽 상태 + 고정 주의문 |

**데이터가 말하지 않는 반도체 물리 메커니즘을 AI가 사실처럼 서술하지 않습니다.**

**외부 LLM API 없이 모든 핵심 기능이 동작합니다.**
`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`가 있을 때만
이미 계산된 수치를 자연어로 요약하는 선택적 카드가 추가됩니다.

---

## 10. 실행 방법

### 원클릭 (Windows)

```cmd
run.cmd
```

### 수동 실행

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

python train_model.py           # 최초 1회, 약 19분 (i5-1130G7 기준)
#   빠른 확인용:  python train_model.py --quick

uvicorn app:app --reload --port 8000
#   또는:  python app.py

python tests/test_pipeline.py   # pytest 없이도 실행 가능
python -m pytest tests -v
```

| URL | 내용 |
|---|---|
| **http://127.0.0.1:8000** | 대시보드 (7개 탭) |
| http://127.0.0.1:8000/docs | Swagger API 문서 |
| http://127.0.0.1:8000/api/health | 헬스체크 |

> 학습 없이 서버를 띄우면 UI가 "모델 미학습"을 표시하고 API는 **503**을 반환합니다.

---

## 11. 화면 구성 (7개 탭)

| 탭 | 기능 |
|---|---|
| **01 Data Explorer** | 데이터 무결성/DOE 구조/컬럼 통계, 인터랙티브 산점도, 주효과 4종, Correlation Heatmap, Parallel Coordinates, Data Observation |
| **02 AI Prediction** | 4변수 → Xj/ΔXj/Rsh 예측, 외삽 경고, 접합 단면도 + **ΔXj 확대 inset**, Local Sensitivity, **Local SHAP**, Insight, 최근접 TCAD run |
| **03 What-If Analysis** | 단일 변수 sweep + **실제 TCAD run 중첩**, 4변수 Process Window Overview |
| **04 AI Process Optimizer** | Target Spec, MODE A/B, 가중치, lock, **Pareto**, Top-10 Recipe, **Nominal vs Robust 재순위** |
| **05 XAI / Explainability** | Permutation Importance + Global SHAP, 방법론, 레이더 비교 |
| **06 Process Robustness** | **실측 산포 레퍼런스**, tolerance preset/개별 조정, Monte-Carlo 분포·수율·Cpk, **산포 기여도**, 모델 vs 공정 불확실성, 경계 clipping 경고 |
| **07 Model Validation** | 3중 검증 지표, Actual vs Predicted / Residual, **전체 모델 leaderboard**, 커버리지, leakage 정책, 심사용 고지문 |

UI는 `tcad_simulator.html`의 Dark Semiconductor Engineering Theme을 계승했습니다.
`tcad_simulator.html`은 **Frontend/UI 레퍼런스로만** 사용했고, 내부 수식·하드코딩 값은 Ground Truth로 쓰지 않았습니다.

---

## 12. 프로젝트 구조

```
chungbuk_ai_semiconductor/
├── app.py                        FastAPI 백엔드 (REST 12개 + 정적 서빙)
├── train_model.py                모델 비교·선택·학습 진입점
├── run.cmd                       Windows 원클릭 실행
├── requirements.txt / README.md
├── core/
│   ├── config.py                 경로 · 변수 정의 · leakage 금지 목록 · 고지문
│   ├── dataset.py                CSV 로딩/검증, DOE 구조, 상관, 주효과
│   ├── modeling.py               모델 zoo, 2중 CV, 쌍 선택, 배포 모델 학습
│   ├── surrogate.py              런타임 추론, 외삽 판정, ΔXj 분해능, 모델 불확실성
│   ├── explain.py                정확 Shapley(2⁴) · Permutation · local sensitivity
│   ├── optimize.py               MODE A/B 탐색, 다목적 스코어, Pareto, 중복/이격
│   ├── whatif.py                 단일 변수 sweep + 실제 TCAD run 오버레이
│   ├── variation.py              **실측 17,000행 계측 → 공정 산포 측정**
│   ├── robust.py                 **Monte-Carlo 수율/Cpk/산포 기여도/robust 재순위**
│   ├── insight.py                3단계 근거 인사이트 생성
│   └── llm.py                    선택적 LLM 서술 (키 없으면 자동 비활성)
├── data/
│   ├── implant_anneal_1000.csv           AI 학습용 Ground Truth (TCAD)
│   └── fab_thickness_profile_17000.csv   실측 산포 레퍼런스 (PECVD 계측)
├── models/surrogate_bundle.joblib        배포 모델 3개 + 전체 리포트
├── outputs/
│   ├── training_report.json      모든 학습 지표의 단일 출처
│   ├── variation_reference.json  실측 산포 측정 결과
│   ├── leaderboard_*.csv         target별 모델 비교표
│   └── train_log.txt
├── static/  index.html · css/app.css · js×9 · vendor/chart.umd.js(오프라인 대비)
├── docs/screenshots/             탭별 스크린샷
└── tests/test_pipeline.py        33개 검증 테스트
```

### 아키텍처

```
Browser (HTML/CSS/JS, Chart.js)
        ↓  REST (JSON)
FastAPI  (app.py)
        ↓
ML Surrogate                  Monte-Carlo Robustness
(models/surrogate_bundle)  ←  (core/robust.py)
        ↓                              ↑
implant_anneal_1000.csv       fab_thickness_profile_17000.csv
  (학습 Ground Truth)            (산포 크기 참조 · 학습 미사용)
```

---

## 13. 검증 (실제 수행 결과)

`python tests/test_pipeline.py` → **33/33 PASS**

* 데이터 무결성 (1000행 · 결측 0 · Full Factorial · ΔXj 항등식)
* **Data leakage 차단** — feature 행렬에 출력 컬럼/`run_id` 절대 미포함
* 배포 모델이 실제 TCAD 1,000 run 재현 (in-sample R² > 0.99)
* 외삽/보간/격자점 3단계 판정
* **SHAP efficiency 공리** Σφ = f(x) − E[f(X)]
* `xj_implant`의 anneal 기여도 ≈ 0
* MODE A 추천값이 CSV 실제 측정값과 일치 / MODE B가 학습 범위 이탈 안 함
* Pareto 집합의 비지배성 / Rsh 제약 / 가중치 반영
* **실측 산포가 assumed가 아닌 measured임** (17,000행 · 1,000 wafer · 6 recipe 검증)
* **tolerance preset 단조성**, custom 플래그, 산포 0 → 수율 100%
* **tolerance를 키우면 수율이 절대 올라가지 않음**
* **산포 기여도 합 = 100%**, Xj 최대 기여 변수 = energy
* **경계 clipping 감지** (내부 조건 미발생 / 경계 조건 발생)
* **Monte-Carlo 샘플이 학습 범위를 절대 벗어나지 않음**
* robust 재순위 정렬 일관성 / 모델 불확실성 양수·출처 라벨
* 전체 REST 엔드포인트 + 정적 파일 + 잘못된 입력 422

브라우저 실측: 7개 탭 전체 렌더링, **콘솔 에러 0건**.

**API 응답 시간 (i5-1130G7, 로컬)**

| 엔드포인트 | 시간 |
|---|---|
| `/api/predict` (SHAP 포함) | ~0.6 s |
| `/api/whatif` (61 pts) | ~0.08 s |
| `/api/optimize` MODE A | ~0.16 s |
| `/api/optimize` MODE B (28,397) | ~2.7 s |
| `/api/robust` (2,500 samples + 분해) | ~2.9 s |
| `/api/variation` | ~0.005 s (캐시) |
| `/api/validation`, `/api/xai/global` | < 0.01 s |

---

## 14. 한계

1. **예측 모델의 학습 데이터가 TCAD 시뮬레이션 결과입니다.**
   이온주입/열처리 공정의 실제 Fab 계측 데이터가 아니며, 장비 산포·웨이퍼 내 균일도·
   챔버 드리프트·계측 오차가 학습 데이터에 포함되어 있지 않습니다.
2. **산포 레퍼런스는 다른 공정(PECVD SiON)의 실측값입니다.**
   크기의 참조 기준일 뿐 이온주입기의 실제 tolerance가 아닙니다.
   Robustness 결과는 "이 정도 산포가 있다면"이라는 **조건부 분석**입니다.
3. **적용 범위는 DOE 상자 내부로 한정됩니다.**
   (dose 5e14–2.5e15, energy 10–50 keV, 900–1100 °C, 5–45 s)
4. **ΔXj의 분해능이 유한합니다.** 파생 ΔXj hold-out RMSE 3.31e-4 um 미만의 확산량은
   모델 오차와 구분되지 않습니다(앱에서 명시 경고).
5. **Group CV 성능이 Random Hold-out보다 낮습니다.** (Xj Final R² 0.9932 vs 0.99999)
6. **단일 물질계/단일 공정 흐름**만 다룹니다. 이온 종, 채널링, tilt/twist,
   전기적 활성화율 등은 데이터에 없어 모델링되지 않았습니다.
7. **생산성 향상·수율 개선 수치를 주장하지 않습니다.** 근거 데이터가 없습니다.
   본 시스템은 TCAD나 Fab 검증을 대체하지 않으며, 추천 recipe는 split 실험 후보입니다.

---

## 15. 확장 경로 (신규 데이터 확보 시)

현재 버전은 **추가 데이터 없이 완결**되어 있습니다. 아래는 데이터가 확보될 경우의 확장안입니다.

| 단계 | 필요 데이터 | 확장 내용 |
|---|---|---|
| 1 | 이온주입기 장비 로그 (dose/energy 실제 산포) | Robustness의 tolerance를 **해당 장비 실측값**으로 교체 → 조건부 분석이 실측 기반 분석으로 승격 |
| 2 | SIMS/SRP 기반 Xj, 4-point probe 기반 Rsh 계측 | GP를 사전(prior)으로 둔 Bayesian calibration으로 TCAD-실측 편차 보정 |
| 3 | (데이터 불필요) | GP 예측 분산 기반 **Active Learning** — 다음에 TCAD를 돌릴 조건을 모델이 제안 |
| 4 | 다중 implant·후속 공정 데이터 | 이온 종·tilt·채널링·Silicide/Contact까지 변수 확장 |
| 5 | MES/EES 연동 | 드리프트 감지 후 재학습 파이프라인 |

3단계(Active Learning)는 **현재 데이터만으로 즉시 구현 가능**하며, 우선 확장 대상입니다.

---

## 16. 고지 (Disclaimer)

> **This AI model is trained on TCAD-generated simulation data.
> Actual semiconductor manufacturing deployment requires additional Fab data and physical validation.**
>
> 본 AI 모델은 TCAD 시뮬레이션으로 생성된 DOE 데이터로 학습되었습니다.
> 실제 반도체 양산 라인 적용을 위해서는 추가적인 Fab 계측 데이터 확보와
> 물리적 검증(Split 실험)이 반드시 필요합니다.
>
> Robustness 분석에 사용된 공정 산포는 **PECVD SiON 증착 공정의 실측값**으로,
> 이온주입 공정의 실제 장비 산포가 아닙니다. 크기의 참조 기준으로만 사용하십시오.

이 문구는 앱 **Model Validation** 탭과 **Process Robustness** 탭, `/api/validation`,
`/api/variation`, `/api/meta` 응답에 모두 포함됩니다.

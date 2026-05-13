# CMAPSS 항공 엔진 예방정비 지능형 대시보드

> **NASA CMAPSS 데이터셋 기반 항공 엔진 잔여 수명(RUL) 예측 및 실시간 상태 모니터링 시스템**  
> 초보 정비사도 직관적으로 활용할 수 있는 AI 통합 예방정비 플랫폼

---

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [데이터셋 소개](#데이터셋-소개)
3. [프로젝트 구조](#프로젝트-구조)
4. [모델 파이프라인](#모델-파이프라인)
5. [대시보드 주요 기능](#대시보드-주요-기능)
6. [설치 및 실행](#설치-및-실행)
7. [기술 스택](#기술-스택)
8. [모델 성능](#모델-성능)
9. [팀 구성](#팀-구성)

---

## 프로젝트 개요

항공 엔진의 **비계획적 고장(Unplanned Failure)** 은 막대한 경제적 손실과 안전사고로 이어질 수 있습니다. 본 프로젝트는 NASA가 제공하는 **CMAPSS(Commercial Modular Aero-Propulsion System Simulation)** 시뮬레이션 데이터를 활용하여 엔진의 **잔여 유효 수명(RUL, Remaining Useful Life)** 을 예측하고, 이를 초보 정비사도 쉽게 활용할 수 있는 인터랙티브 대시보드로 제공합니다.

### 핵심 목표

| 목표 | 내용 |
|------|------|
| **예측 정확도** | RMSE 기준 최저 13.10 달성 (FD003, XGBoost) |
| **현장 적용성** | ML 전문 지식 없는 정비사도 사용 가능한 UI/UX |
| **실시간 모니터링** | 전체 엔진 상태를 한눈에 파악하는 대시보드 |
| **AI 연동** | Gemini API 기반 자연어 정비 상담 챗봇 |

---

## 📊 데이터셋 소개

### CMAPSS (Commercial Modular Aero-Propulsion System Simulation)

NASA Glenn Research Center에서 제공하는 항공기 터보팬 엔진 열화(Degradation) 시뮬레이션 데이터셋입니다.

```
CMAPSSData/
├── test_FD001.csv              # 테스트 데이터 (FD001)
├── test_FD001_preprocessed.csv # 전처리된 테스트 데이터
├── test_FD002.csv
├── test_FD002_preprocessed.csv
├── test_FD003.csv
├── test_FD003_preprocessed.csv
├── test_FD004.csv
├── test_FD004_preprocessed.csv
├── RUL_FD001.csv               # 실제 RUL 라벨
├── RUL_FD002.csv
├── RUL_FD003.csv
├── RUL_FD004.csv
├── preprocessed_test_FD001.csv # 운항조건 클러스터링 포함 전처리본
├── preprocessed_test_FD002.csv
├── preprocessed_test_FD003.csv
└── preprocessed_test_FD004.csv
```

### 데이터셋 구성

| 데이터셋 | 학습 엔진 수 | 테스트 엔진 수 | 운항 조건 | 결함 모드 |
|---------|------------|--------------|---------|---------|
| FD001 | 100 | 100 | 1 (Sea Level) | HPC 압축기 열화 |
| FD002 | 260 | 259 | 6 (복합 조건) | HPC 압축기 열화 |
| FD003 | 100 | 100 | 1 (Sea Level) | HPC + Fan 열화 |
| FD004 | 249 | 248 | 6 (복합 조건) | HPC + Fan 열화 |

### 센서 구성 (21개)

| 센서 코드 | 명칭 | 단위 | 센서 코드 | 명칭 | 단위 |
|---------|------|------|---------|------|------|
| s_1 (T2) | 팬 입구 온도 | °R | s_12 (phi) | 연료/압력 비율 | pps/psia |
| s_2 (T24) | LPC 출구 온도 | °R | s_13 (NRf) | 보정 팬 속도 | rpm |
| s_3 (T30) | HPC 출구 온도 | °R | s_14 (NRc) | 보정 코어 속도 | rpm |
| s_4 (T50) | LPT 출구 온도 | °R | s_15 (BPR) | 바이패스 비율 | - |
| s_7 (P30) | HPC 출구 압력 | psia | s_17 (htBleed) | 블리드 엔탈피 | - |
| s_8 (Nf) | 팬 속도 | rpm | s_20 (W31) | HPT 냉각 블리드 | lbm/s |
| s_9 (Nc) | 코어 속도 | rpm | s_21 (W32) | LPT 냉각 블리드 | lbm/s |
| s_11 (Ps30) | HPC 출구 정적압력 | psia | | | |

> FD001/003 기준 유효 센서 14개, FD002 20개, FD003/004 15개 (상수값 센서 제외)

---

## 프로젝트 구조

```
project/
│
├── 📁 CMAPSSData/                    # 원본 및 전처리 데이터
│   ├── test_FD00{1~4}.csv            # 원본 테스트 데이터
│   ├── test_FD00{1~4}_preprocessed.csv  # 전처리 데이터 (정규화)
│   ├── preprocessed_test_FD00{1~4}.csv  # 운항조건 클러스터링 포함
│   └── RUL_FD00{1~4}.csv            # 실제 RUL 라벨
│
├── 📁 notebooks/                     # 분석 노트북
│   ├── EDA_FD001~4.ipynb            # 탐색적 데이터 분석
│   ├── Preprocessing_FD001~4.ipynb  # 전처리 파이프라인
│   ├── ML_FD001~4.ipynb             # 머신러닝 모델 실험
│   └── DL_FD001~4.ipynb             # 딥러닝 모델 실험
│
├── 📁 saved_models/                  # 학습 완료 모델 및 결과
│   ├── model_XGBoost_FD00{1~4}.pkl  # XGBoost 최종 모델
│   ├── model_LightGBM_FD00{1~4}.pkl # LightGBM 최종 모델
│   ├── summary_XGBoost_FD00{1~4}.csv  # 예측 결과 요약
│   ├── summary_LightGBM_FD00{1~4}.csv
│   ├── history_XGBoost_FD00{1~4}.csv  # 학습 이력
│   └── history_LightGBM_FD00{1~4}.csv
│
├── 📄 app.py                         # Streamlit 대시보드 메인
├── 📄 main.py                        # 진입점
├── 📄 build_db1.py                   # DuckDB 원본 데이터 구축
├── 📄 build_db2.py                   # DuckDB 전처리 데이터 추가
├── 📄 cmapss.db                      # DuckDB 데이터베이스 (LFS)
├── 📄 engine_diagram.png             # 엔진 단면도 이미지
├── 📄 requirements.txt               # Python 패키지 목록
├── 📄 runtime.txt                    # Python 버전 명시
└── 📄 .streamlit/secrets.toml        # API 키 (로컬 전용, gitignore)
```

---

## 모델 파이프라인

### 1. 전처리 (Preprocessing)

```
원본 데이터
    │
    ├─ 상수 센서 제거 (분산 ≈ 0인 센서 제외)
    │   └─ FD001: s_1, s_5, s_6, s_10, s_16, s_18, s_19 제거
    │
    ├─ 운항 조건 정규화 (FD002/004)
    │   └─ KMeans 클러스터링 (k=6) → op_cluster 컬럼 생성
    │   └─ 클러스터별 MinMaxScaler 정규화
    │
    ├─ 피쳐 엔지니어링
    │   └─ 롤링 평균/표준편차 (window=5, 10)
    │   └─ 직전 사이클 대비 변화량
    │
    └─ RUL 레이블 생성
        └─ Piecewise Linear Degradation (최대 RUL 캡 적용)
```

### 2. 모델 실험

#### 머신러닝 (ML)
- **XGBoost**: 그래디언트 부스팅 기반, FD002/003/004 최우수
- **LightGBM**: 경량화 그래디언트 부스팅, FD001 최우수
- **Random Forest**: 앙상블 기법
- **Ridge / Lasso**: 선형 회귀 기반 베이스라인

#### 딥러닝 (DL)
- **LSTM**: 시계열 장기 의존성 학습
- **CNN-LSTM**: 합성곱 + 순환 신경망 하이브리드
- **Transformer**: 어텐션 메커니즘 기반

#### 하이퍼파라미터 최적화
- **Optuna** 기반 베이지안 최적화
- 5-Fold Cross Validation

### 3. 평가 지표

```python
# RMSE (Root Mean Square Error)
RMSE = sqrt(mean((y_true - y_pred)²))

# NASA Score (비대칭 손실 함수 — 조기 예측 패널티 강화)
score = sum(exp(-d/13) - 1  if d < 0  else  exp(d/10) - 1)
# d = y_pred - y_true
```

> NASA Score는 늦은 예측(과다 예측)을 조기 예측보다 더 강하게 패널티를 부여하는 비대칭 손실 함수입니다.

---

## 대시보드 주요 기능

### 탭 1: 전체 현황
- **엔진 현황 요약 카드**: 즉시 점검 / 점검 예약 / 정상 운행 현황을 색상 코딩으로 표시
- **잔여 수명 분포 히스토그램**: 전체 엔진 수명 분포 시각화
- **엔진별 상태 카드**: 상태 필터 / 정렬 / 엔진 번호 범위 슬라이더 필터 제공
- **상세 보기 Expander**: 부품별 이상 징후 및 권고 조치 즉시 확인

### 탭 2: 엔진 상세
- **4개 핵심 지표 카드**: 엔진 번호 / 종합 상태(색상 배경) / 총 사이클 / 잔여 수명
- **잔여 수명 게이지**: 데이터셋 최대 수명 대비 비율 시각화
- **부품별 이상 징후 추정**: HPC / 팬 시스템 / LPC / 터빈 / 코어 이상도 (%)
  - Z-score 기반 전체 데이터셋 대비 상대적 위험도 산출
  - 기준: 0~50% 정상 / 51~65% 관찰 / 66%+ 주의
- **부품별 센서 궤적**: 롤링 평균 추이선 포함 전체 운전 이력 그래프
- **권고 조치**: 주요 이상 부품에 특화된 점검 가이드 자동 생성
- **전체 엔진 RUL 비교 바차트**: 위험/주의/정상 색상 코딩

### 탭 3: 센서 분석
- **엔진 단면도 + 실시간 상태 표시**: 부품별 이상 여부를 이미지 오버레이로 표시
- **현재 주요 센서 지표**: 최신 사이클 기준 8개 센서 게이지 카드
  - 이상 감지: 최근 10사이클 기울기 + 전체 분포 극단값 결합 알고리즘
  - FD002/004: 운항 조건(op_cluster)별 분리 분석으로 오탐 방지
- **핵심 센서 N개 선택 버튼**: 데이터셋별 유효 센서 전체 자동 선택
- **센서 추세 분석**: 원시 측정값 / 운항 조건 보정 후 추세 2개 차트
- **수명 감소 연관 핵심 센서 TOP 10**: 절댓값 상관계수 기반 막대차트
  - 방향 표시: 수명 줄면 상승(📈) / 하강(📉)
  - 연관 강도: 0.7+ 빨강 / 0.4~0.7 노랑 / ~0.4 초록
- **SHAP 기반 수명 예측 근거**: 센서별 예측 영향도 waterfall 차트 및 자연어 설명

### 탭 4: 점검 이력 & 즉시접수
- **점검 이력 요약**: 총 점검 횟수 / 마지막 점검일 / 정기·긴급점검 횟수
- **타임라인 형식 이력 표시**: 일자 / 유형 배지 / 점검 내용 / 담당자
- **점검 기록 추가 폼**: 유형 선택 + 담당자 + 내용 입력 → 세션 저장
- **AI 점검 이력 분석**: Gemini API가 이력 패턴 분석 및 반복 문제 감지
- **즉시 점검 접수**: 잔여 수명 30사이클 미만 엔진 일괄/개별 접수
- **CSV 다운로드**: 접수 이력 엑셀 파일로 저장

### 사이드바: AI 정비 상담 챗봇
- **Gemini 2.5 Flash** 기반 자연어 대화
- 현재 선택된 엔진의 실제 센서값 + RUL 데이터를 프롬프트에 자동 주입
- 초보 정비사 맞춤 응답 규칙:
  - ML 기술 용어(RMSE, MAE 등) 사용 금지
  - 센서 코드 대신 한글 명칭 사용
  - RUL을 일 수/운항 횟수로 자동 환산
  - 모든 답변에 권고 행동 포함
- 빠른 질문 버튼 4개 제공

---

## 설치 및 실행

### 요구사항
- Python 3.12+
- 4GB RAM 이상

### 로컬 실행

```bash
# 1. 저장소 클론
git clone https://github.com/kt01054315893-tech/project.git
cd project

# 2. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 데이터베이스 구축 (최초 1회)
python build_db1.py   # 원본 데이터 로드
python build_db2.py   # 전처리 데이터 추가

# 5. API 키 설정
mkdir -p .streamlit
echo 'GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"' > .streamlit/secrets.toml

# 6. 대시보드 실행
streamlit run app.py
```

### Streamlit Cloud 배포

1. GitHub 저장소 연결
2. **Settings → Secrets** 에 추가:
```toml
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"
```
3. `requirements.txt` 의존성 자동 설치 후 배포 완료

---

## 기술 스택

### 데이터 처리 & 모델링
| 라이브러리 | 버전 | 용도 |
|---------|------|------|
| pandas | 2.x | 데이터 처리 |
| numpy | 1.x | 수치 연산 |
| scikit-learn | 1.x | 전처리, 평가 지표 |
| XGBoost | 2.x | RUL 예측 (주 모델) |
| LightGBM | 4.x | RUL 예측 (주 모델) |
| shap | 0.x | 모델 설명 가능성 (SHAP) |
| optuna | 3.x | 하이퍼파라미터 최적화 |

### 데이터베이스 & 스토리지
| 도구 | 용도 |
|-----|------|
| DuckDB | 경량 인메모리 분석 데이터베이스 |
| Git LFS | 대용량 파일(cmapss.db, CSV) 버전 관리 |

### 대시보드 & AI
| 라이브러리 | 용도 |
|---------|------|
| Streamlit | 웹 대시보드 프레임워크 |
| Plotly | 인터랙티브 시각화 |
| pydantic-ai | AI 에이전트 프레임워크 |
| Google Gemini 2.5 Flash | 자연어 정비 상담 LLM |
| nest-asyncio | 비동기 실행 환경 호환 |

---

## 모델 성능

### 최종 선정 모델 성능 (테스트셋 기준)

| 데이터셋 | 최적 모델 | RMSE | MAE | NASA Score |
|---------|---------|------|-----|-----------|
| **FD001** | LightGBM | **14.01** | 10.69 | 185.5 |
| **FD002** | XGBoost | **23.51** | 15.99 | 4,043.55 |
| **FD003** | XGBoost | **13.10** | 10.01 | 296.94 |
| **FD004** | XGBoost | **24.89** | 17.52 | 3,678.48 |

> FD002/004의 NASA Score가 높은 이유: 6가지 복합 운항 조건으로 인한 예측 난이도 증가

### 이상 감지 알고리즘 (센서 분석)

```python
# 1단계: 상수 센서 필터링
if (arr.max() - arr.min()) < abs(arr.mean()) * 0.001:
    return 0.0  # 변화 없는 센서 제외

# 2단계: 최근 10사이클 기울기 (전체 std 대비)
slope_score = min(1.0, |slope| / global_std / 0.5) * 0.6

# 3단계: 현재값의 전체 분포 내 극단 위치
extremity = |pct_rank - 0.5| * 2 * 0.4

# 4단계: RUL 잔여 수명 보정
rul_score = max(0, (0.2 - rul_ratio) / 0.2 * 0.8) if rul_ratio < 0.2 else 0

# 최종 이상 점수 = 센서 70% + RUL 보정 30%
anomaly_score = (slope_score + extremity) * 0.7 + rul_score * 0.3
```

## 라이선스

본 프로젝트는 학술 목적으로 제작되었습니다.  
CMAPSS 데이터셋은 NASA Glenn Research Center에서 공개 제공합니다.

> 참고: A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA

---

<div align="center">

**✈️ CMAPSS 항공 엔진 예방정비 지능형 대시보드**  
*Sparta Coding Club AI 트랙 팀 프로젝트*

</div>

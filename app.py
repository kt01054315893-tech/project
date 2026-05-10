import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import duckdb
import re
import asyncio
from datetime import datetime
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
 
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
 
# =========================================================
# 1. 페이지 설정 & 전역 CSS
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="엔진 상태 관리 시스템",
    page_icon="✈️"
)
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif;
}
 
/* ── 전체 배경 ── */
.stApp { background-color: #0d1117; }
 
/* ── 사이드바 ── */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e6edf3;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}
 
/* ── 탭 스타일 ── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #161b22;
    border-bottom: 1px solid #21262d;
    padding: 0 8px;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #8b949e;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 12px 20px;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #58a6ff !important;
    background-color: transparent !important;
}
 
/* ── 상태 카드 ── */
.status-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.status-card:hover { border-color: #30363d; }
.status-card.danger  { border-left: 4px solid #f85149; }
.status-card.warning { border-left: 4px solid #d29922; }
.status-card.normal  { border-left: 4px solid #3fb950; }
 
.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.card-engine-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 600;
    color: #e6edf3;
}
.badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}
.badge.danger  { background: rgba(248,81,73,0.15);  color: #f85149; }
.badge.warning { background: rgba(210,153,34,0.15); color: #d29922; }
.badge.normal  { background: rgba(63,185,80,0.15);  color: #3fb950; }
 
.rul-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e6edf3;
    line-height: 1;
    margin-bottom: 4px;
}
.rul-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-bottom: 10px;
}
.progress-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-bottom: 8px;
}
.progress-bar-fill {
    height: 6px;
    border-radius: 4px;
    transition: width 0.5s ease;
}
.action-text {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 6px;
}
 
/* ── 상단 요약 바 ── */
.summary-bar {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 24px;
    display: flex;
    gap: 40px;
    align-items: center;
    margin-bottom: 24px;
}
.summary-item-label { font-size: 0.75rem; color: #8b949e; margin-bottom: 2px; }
.summary-item-value { font-size: 1.3rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.sv-danger  { color: #f85149; }
.sv-warning { color: #d29922; }
.sv-normal  { color: #3fb950; }
 
/* ── 섹션 헤더 ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #58a6ff;
    margin: 24px 0 12px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}
 
/* ── 게이지 컨테이너 ── */
.gauge-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}
.gauge-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 16px;
}
.gauge-name { font-size: 0.78rem; color: #8b949e; margin-bottom: 4px; }
.gauge-val  { font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; color: #e6edf3; margin-bottom: 8px; }
 
/* ── 챗봇 ── */
.chat-quick-btn {
    background: #21262d;
    border: 1px solid #30363d;
    border-radius: 20px;
    color: #c9d1d9;
    font-size: 0.8rem;
    padding: 6px 14px;
    margin: 3px;
    cursor: pointer;
    display: inline-block;
}
 
/* ── 알림 배너 ── */
.alert-banner {
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 16px;
    font-size: 0.875rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.alert-danger  { background: rgba(248,81,73,0.1);  border: 1px solid rgba(248,81,73,0.3);  color: #f85149; }
.alert-warning { background: rgba(210,153,34,0.1); border: 1px solid rgba(210,153,34,0.3); color: #d29922; }
 
/* ── 이력 테이블 ── */
.history-row {
    display: grid;
    grid-template-columns: 100px 100px 1fr 80px;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.85rem;
    color: #c9d1d9;
    align-items: center;
}
.history-row:last-child { border-bottom: none; }
.history-date { font-family: 'JetBrains Mono', monospace; color: #8b949e; }
.history-type-badge {
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 12px;
    display: inline-block;
}
.type-regular  { background: rgba(88,166,255,0.15); color: #58a6ff; }
.type-urgent   { background: rgba(248,81,73,0.15);  color: #f85149; }
.type-check    { background: rgba(63,185,80,0.15);  color: #3fb950; }
 
/* ── Metric 스타일 재정의 ── */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"] { color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace; font-size: 1.4rem !important; }
 
/* ── 공통 카드 컨테이너 ── */
.info-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 20px;
}
 
/* ── 버튼 ── */
.stButton > button {
    background: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 0.85rem;
    padding: 6px 14px;
}
.stButton > button:hover {
    background: #30363d;
    border-color: #58a6ff;
    color: #58a6ff;
}
 
/* ── 입력 필드 ── */
.stSelectbox > div, .stNumberInput > div { background: #161b22 !important; }
</style>
""", unsafe_allow_html=True)
 
# =========================================================
# 2. 상수 & 매핑
# =========================================================
os.environ["GOOGLE_API_KEY"] = "Your_Key"
 
# 센서 코드 → 한글 명칭, 단위
SENSOR_META = {
    "s_1":  {"name": "팬 입구 온도",            "symbol": "T2",        "unit": "°R"},
    "s_2":  {"name": "LPC 출구 온도",           "symbol": "T24",       "unit": "°R"},
    "s_3":  {"name": "HPC 출구 온도",           "symbol": "T30",       "unit": "°R"},
    "s_4":  {"name": "LPT 출구 온도",           "symbol": "T50",       "unit": "°R"},
    "s_5":  {"name": "팬 입구 압력",            "symbol": "P2",        "unit": "psia"},
    "s_6":  {"name": "바이패스 압력",           "symbol": "P15",       "unit": "psia"},
    "s_7":  {"name": "HPC 출구 압력",           "symbol": "P30",       "unit": "psia"},
    "s_8":  {"name": "팬 속도",                 "symbol": "Nf",        "unit": "rpm"},
    "s_9":  {"name": "코어 속도",               "symbol": "Nc",        "unit": "rpm"},
    "s_10": {"name": "엔진 압력비",             "symbol": "epr",       "unit": "-"},
    "s_11": {"name": "HPC 출구 정적압력",       "symbol": "Ps30",      "unit": "psia"},
    "s_12": {"name": "연료/압력 비율",          "symbol": "phi",       "unit": "pps/psia"},
    "s_13": {"name": "보정 팬 속도",            "symbol": "NRf",       "unit": "rpm"},
    "s_14": {"name": "보정 코어 속도",          "symbol": "NRc",       "unit": "rpm"},
    "s_15": {"name": "바이패스 비율",           "symbol": "BPR",       "unit": "-"},
    "s_16": {"name": "버너 연료공기비",         "symbol": "farB",      "unit": "-"},
    "s_17": {"name": "블리드 엔탈피",           "symbol": "htBleed",   "unit": "-"},
    "s_18": {"name": "요구 팬 속도",            "symbol": "Nf_dmd",    "unit": "rpm"},
    "s_19": {"name": "요구 코어 속도",          "symbol": "PCNfR_dmd", "unit": "rpm"},
    "s_20": {"name": "HPT 냉각 블리드",         "symbol": "W31",       "unit": "lbm/s"},
    "s_21": {"name": "LPT 냉각 블리드",         "symbol": "W32",       "unit": "lbm/s"},
}

# 데이터셋별 유효 센서
USEFUL_SENSORS = {
    "FD001": [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21],
    "FD002": [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21],
    "FD003": [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21],
    "FD004": [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21],
}
 
CLUSTER_MAP = {
    0: "최대 고도 고속 (42k, 0.84M)", 1: "지상 및 이륙 (0ft, 0M)",
    2: "중고도 비행 (20k, 0.7M)",    3: "중고도 가변 (25k, 0.6M)",
    4: "저고도 접근 (10k, 0.25M)",   5: "일반 순항 (35k, 0.84M)",
}
 
# 부품 이름 매핑 (센서 → 부품)
PART_MAP = {
    "HPC": ["s_3", "s_7", "s_11", "s_12"],
    "팬 시스템": ["s_1", "s_5", "s_8", "s_13", "s_15"],
    "LPC": ["s_2", "s_14"],
    "터빈 (LPT/HPT)": ["s_4", "s_17", "s_20", "s_21"],
    "코어": ["s_9", "s_10"],
}
 
# 점검 이력 (세션 상태로 관리)
MOCK_HISTORY = [
    {"date": "2025.03.15", "type": "정기점검", "note": "팬 블레이드 마모 확인 후 교체 완료",         "by": "김정비"},
    {"date": "2025.01.20", "type": "일상점검", "note": "전체 센서 정상 범위 내 확인, 윤활유 보충",  "by": "이정비"},
    {"date": "2024.11.05", "type": "긴급점검", "note": "HPC 출구 온도 상승 감지 → 압축기 세정 실시","by": "박정비"},
    {"date": "2024.08.12", "type": "정기점검", "note": "LPT 냉각 블리드 유량 점검, 이상 없음",      "by": "김정비"},
]
 
# =========================================================
# 3. 유틸 함수
# =========================================================
def get_rul_status(rul: int):
    """RUL → (상태명, 색상 클래스, 배지, 아이콘, 권고문)"""
    if rul < 30:
        return "위험", "danger", "#f85149", "🔴", f"즉시 점검 필요 — 잔여 {rul} 사이클"
    elif rul < 60:
        return "주의", "warning", "#d29922", "🟡", f"30사이클 이내 점검 예약 권고"
    else:
        return "정상", "normal", "#3fb950", "🟢", "정상 운영 가능"
 
def _score_single_series(arr: np.ndarray, window: int) -> float:
    """
    단일 시계열에서 센서 이상 점수(0~1) 계산.

    상수 센서 처리: 전체 std가 매우 작으면(변화 없는 센서) 추세 점수를 0으로 처리.
    → 변화 없는 센서가 "이상"으로 잡히거나 점수를 희석시키는 것을 방지.

    신호 1 (60%): 최근 window 사이클의 선형 기울기 / 전체 std
    신호 2 (40%): 최근값의 전체 분포 내 극단 위치 (0=중간, 1=극단)
    """
    n = min(window, len(arr))
    if n < 3:
        return 0.0

    global_std = arr.std()
    global_range = arr.max() - arr.min()

    # 상수 센서 판별: range가 전체 평균의 0.1% 미만이면 변화 없는 센서로 간주
    if global_range < (abs(arr.mean()) * 0.001 + 1e-9):
        return 0.0  # 상수 센서 → 점수 기여 없음

    recent = arr[-n:]
    x = np.arange(n)
    slope = np.polyfit(x, recent, 1)[0]
    norm_slope = abs(slope) / (global_std + 1e-9)
    pct_rank = float((arr < recent[-1]).mean())
    extremity = abs(pct_rank - 0.5) * 2
    slope_score = min(1.0, norm_slope / 0.5)
    return slope_score * 0.6 + extremity * 0.4


def detect_sensor_status(
    series: pd.Series,
    window: int = 10,
    cluster_series: pd.Series = None,
    pred_rul: int = None,
    dataset_max_rul: int = None,
):
    """
    센서 이상 감지 — 센서 추세 + RUL 잔여 수명을 결합한 최종 점수.

    [FD001/003 - 단일 조건]
      cluster_series=None → raw 전체 시계열로 추세 계산

    [FD002/004 - 복합 조건]
      cluster_series=op_cluster → 마지막 운항 조건 내 데이터만 추출해 추세 계산

    [RUL 보정]
      pred_rul과 dataset_max_rul이 주어지면 수명 잔여 비율을 이상 점수에 추가 반영.
      → 잔여 수명 20% 이하: 센서 점수와 관계없이 최소 "관찰" 보장
      → 잔여 수명 10% 이하: 최소 "이상" 보장
      이를 통해 "센서는 조용하지만 수명 말기" 케이스를 정확히 포착.

    반환: (color, status_label, trend_pct)
    """
    if len(series) < 3:
        return "#8b949e", "데이터 부족", 0

    # 클러스터 분리 (FD002/004)
    if cluster_series is not None and len(cluster_series) == len(series):
        last_cluster = cluster_series.iloc[-1]
        mask = (cluster_series == last_cluster)
        arr = series[mask].values
        if len(arr) < 3:
            arr = series.values
    else:
        arr = series.values

    sensor_score = _score_single_series(arr, window)

    # RUL 기반 보정 점수 계산
    rul_score = 0.0
    if pred_rul is not None and dataset_max_rul and dataset_max_rul > 0:
        rul_ratio = pred_rul / dataset_max_rul        # 0(위험) ~ 1(안전)
        # 수명이 적을수록 rul_score 증가: 20%→0.3, 10%→0.6, 5%→0.8
        rul_score = max(0.0, min(1.0, (0.2 - rul_ratio) / 0.2 * 0.8)) if rul_ratio < 0.2 else 0.0

    # 최종 점수: 센서 70% + RUL 30% (RUL 정보 없으면 센서만)
    if pred_rul is not None:
        anomaly_score = sensor_score * 0.7 + rul_score * 0.3
    else:
        anomaly_score = sensor_score

    trend_pct = int(anomaly_score * 100)

    if anomaly_score > 0.55:
        return "#f85149", "⚠ 이상", trend_pct
    elif anomaly_score > 0.30:
        return "#d29922", "관찰", trend_pct
    else:
        return "#3fb950", "정상", trend_pct
 
 
def rul_to_period(rul: int) -> str:
    """RUL 사이클 → 대략적 기간 (1사이클 ≈ 1일 운항 가정)"""
    if rul < 7:   return f"약 {rul}일"
    if rul < 30:  return f"약 {rul // 7}주"
    if rul < 365: return f"약 {rul // 30}개월"
    return f"약 {rul // 365}년 이상"
 
def calculate_nasa_score(actual, pred):
    d = pred - actual
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
 
def sensor_label(s: str) -> str:
    """s_3 → 'HPC 출구 온도 (T30)'"""
    m = SENSOR_META.get(s)
    if m:
        return f"{m['name']} ({m['symbol']})"
    return s
 

import plotly.graph_objects as go
from PIL import Image

def draw_engine_monitor(sensor_status_dict):
    """
    sensor_status_dict: {'LPC': '#3fb950', 'HPC': '#d29922', ...} 형태의 색상 코드 전달
    """
    try:
        img = Image.open("engine_diagram.png")
    except:
        # 이미지가 없을 경우 빈 배경 처리 (에러 방지)
        st.error("engine_diagram.png 파일을 찾을 수 없습니다.")
        return go.Figure()

    # 부위별 좌표 정의 (0~1 사이 비율)
    # 실제 제공된 엔진 이미지 구조에 따른 대략적 위치
    component_locs = {
        'Fan':      {'x': 0.27, 'y': 0.50, 'label': '팬 (Fan)'},
        'LPC':      {'x': 0.37, 'y': 0.50, 'label': '저압 압축기 (LPC)'},
        'HPC':      {'x': 0.48, 'y': 0.50, 'label': '고압 압축기 (HPC)'},
        'Combustor':{'x': 0.58, 'y': 0.50, 'label': '연소기 (Combustor)'},
        'HPT':      {'x': 0.75, 'y': 0.50, 'label': '고압 터빈 (HPT)'},
        'LPT':      {'x': 0.68, 'y': 0.50, 'label': '저압 터빈 (LPT)'},
    }

    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=img,
            xref="x", yref="y",
            x=0, y=1,
            sizex=1, sizey=1,
            sizing="stretch",
            layer="below"
        )
    )

    for comp, color in sensor_status_dict.items():
        if comp in component_locs:
            loc = component_locs[comp]
            fig.add_trace(go.Scatter(
                x=[loc['x']], y=[loc['y']],
                mode="markers",
                marker=dict(size=35, color=color, opacity=0.7,
                            line=dict(width=3, color='white')),
                name=loc['label'],
                hovertext=f"<b>{loc['label']}</b><br>상태: {color}",
                hoverinfo="text"
            ))

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False
    )
    return fig

# =========================================================
# 4. DB / 파일 로드
# =========================================================
@st.cache_resource
def get_connection():
    # 1. 파일명 정의
    db_file = 'cmapss.db'
    
    # 2. 파일이 존재하는지 먼저 확인
    if not os.path.exists(db_file):
        # 만약 배포 환경에서 경로가 꼬였다면 현재 경로의 파일들을 출력해서 디버깅
        st.error(f"DB 파일을 찾을 수 없습니다. 현재 디렉토리 파일: {os.listdir('.')}")
        return None
        
    try:
        # 파일이 존재할 때만 연결 시도
        return duckdb.connect(db_file, read_only=True)
    except Exception as e:
        st.error(f"DB 연결 오류: {e}")
        return None
    
@st.cache_data
def load_summary(subset: str, model_name: str):
    path = f"saved_models/summary_{model_name}_{subset}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'unit_nr' not in df.columns:
        df['unit_nr'] = df.index + 1
    return df
 
@st.cache_data
def load_history_csv(subset: str, model_name: str):
    path = f"saved_models/history_{model_name}_{subset}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None
 
def load_sensor_data(table_name: str, unit_nr: int):
    con = get_connection()
    q = f"SELECT * FROM {table_name} WHERE unit_nr = {unit_nr} ORDER BY time_cycles"
    return con.execute(q).df()
 
def load_all_summary_best(subset: str):
    """자동으로 최적 모델 summary 로드 (RMSE 기준)"""
    best_model_map = {"FD001": "LightGBM", "FD002": "XGBoost", "FD003": "XGBoost", "FD004": "XGBoost"}
    model = best_model_map.get(subset, "XGBoost")
    df = load_summary(subset, model)
    return df, model

def load_my_model(model_path='model.pkl'):
    # 프로젝트 루트에 model.pkl이 있는지 확인
    if not os.path.exists(model_path):
        return None
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        # 로드 실패 시 에러 내용을 Streamlit에 출력하지 않고 None 반환
        return None
 
# =========================================================
# 5. AI 에이전트 (개선된 시스템 프롬프트)
# =========================================================
async def get_engine_data(ctx: RunContext[None], subset: str, unit_nr: int):
    """특정 엔진의 최신 센서 및 잔여수명 정보를 DB에서 조회합니다."""
    try:
        table_name = f"test_{subset.lower()}"
        con = duckdb.connect('cmapss.db', read_only=True)
        query = f"SELECT * FROM {table_name} WHERE unit_nr = {unit_nr} ORDER BY time_cycles DESC LIMIT 1"
        df = con.execute(query).df()
        con.close()
        if df.empty:
            return f"엔진 #{unit_nr} 데이터를 찾을 수 없습니다."
        return df.to_dict(orient='records')[0]
    except Exception as e:
        return f"DB 조회 오류: {str(e)}"
 
@st.cache_resource
def load_ai_agent():
    try:
        model = GoogleModel('gemini-2.5-flash')
        agent = Agent(
            model=model,
            system_prompt="""
당신은 항공 엔진 예방정비 전문 AI 어시스턴트입니다.
초보 정비사가 쉽게 이해할 수 있도록 도와주세요.
 
[엄격한 답변 규칙]
1. RMSE, MAE, NASA Score, Loss, pred_rul, actual_rul 같은 기술 용어는 절대 사용하지 마세요.
2. 센서 코드(s_2, s_3 등)를 그대로 쓰지 말고 항상 한글 명칭으로 설명하세요.
   예: s_3 → 'HPC 출구 온도', s_7 → 'HPC 출구 압력'
3. RUL 수치는 반드시 기간으로 환산해서 말하세요.
   예: "잔여 수명 45사이클 = 약 1.5개월"
4. 모든 답변 끝에 "✅ 권고 행동:" 한 줄을 추가하세요.
5. 불확실할 경우 "정밀 점검을 권고합니다"로 마무리하세요.
6. 전문 용어 대신 정비사가 이해할 수 있는 일상 언어를 사용하세요.
7. 온도, 압력, 속도 이상 시 어떤 부품을 확인해야 하는지 구체적으로 안내하세요.
 
[데이터 해석 기준]
- 잔여수명 < 30사이클: 즉시 점검 필요 (위험)
- 잔여수명 30~60사이클: 조속한 점검 예약 (주의)
- 잔여수명 > 60사이클: 정상 운행 가능
""",
        )
        agent.tool(get_engine_data)
        return agent
    except Exception as e:
        st.error(f"AI 로드 실패: {e}")
        return None
 
async def run_chat(agent, subset: str, unit_nr: int, query: str) -> str:
    try:
        ctx = f"현재 사용자가 보고 있는 엔진: {subset} 데이터셋의 {unit_nr}번 엔진. 질문: {query}"
        result = await agent.run(ctx)
        return result.data if hasattr(result, 'data') else result.output
    except Exception as e:
        return f"❌ AI 오류: {str(e)}"
 
# =========================================================
# 6. 사이드바
# =========================================================
with st.sidebar:
    st.markdown("### ✈️ 엔진 상태 관리 시스템")
    st.markdown("---")
 
    # 내부적으로만 사용 (정비사에게 노출 최소화)
    subset_choice = st.selectbox(
        "운항 데이터셋",
        ["FD001", "FD002", "FD003", "FD004"],
        help="FD001/003: 단일 조건 | FD002/004: 복합 조건"
    )
    unit_id = st.number_input("엔진 번호", min_value=1, value=1, step=1)
 
    st.markdown("---")
    st.markdown("### 🤖 AI 정비 상담")
 
    # 추천 질문 버튼
    quick_questions = [
        "지금 당장 점검해야 할 엔진이 있나요?",
        f"엔진 #{unit_id}의 상태를 쉽게 설명해주세요",
        f"엔진 #{unit_id}에서 주의해야 할 센서는?",
        "언제 다음 점검을 해야 하나요?",
    ]
 
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "quick_q" not in st.session_state:
        st.session_state.quick_q = None
 
    st.markdown("**빠른 질문:**")
    for q in quick_questions:
        if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
            st.session_state.quick_q = q
 
    st.markdown("---")
 
    # 채팅 이력
    chat_container = st.container(height=320)
    for msg in st.session_state.messages:
        chat_container.chat_message(msg["role"]).write(msg["content"])
 
    # 입력창 (일반 입력 또는 빠른 질문)
    prompt = st.chat_input("질문을 입력하세요")
    if st.session_state.quick_q:
        prompt = st.session_state.quick_q
        st.session_state.quick_q = None
 
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_container.chat_message("user").write(prompt)
 
        ai_agent = load_ai_agent()
        if ai_agent:
            with chat_container.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        resp = loop.run_until_complete(run_chat(ai_agent, subset_choice, unit_id, prompt))
                        loop.close()
                    except Exception as e:
                        resp = f"❌ 오류: {e}"
                    st.write(resp)
            st.session_state.messages.append({"role": "assistant", "content": resp})
 
# =========================================================
# 7. 메인 탭 구성
# =========================================================
tab_overview, tab_engine, tab_sensor, tab_history, tab_scenario = st.tabs([
    "🏠 전체 현황",
    "🔍 엔진 상세",
    "📡 센서 분석",
    "📋 점검 이력",
    "🚨 운영 시나리오",
])
 
# =========================================================
# 탭 1: 전체 현황
# =========================================================
with tab_overview:
    summary_df, best_model = load_all_summary_best(subset_choice)
 
    if summary_df is not None:
        latest = summary_df.drop_duplicates('unit_nr', keep='last')
        n_danger  = int((latest['pred_rul'] < 30).sum())
        n_warning = int(((latest['pred_rul'] >= 30) & (latest['pred_rul'] < 60)).sum())
        n_normal  = int((latest['pred_rul'] >= 60).sum())
        total = len(latest)
 
        # ── 알림 배너 ──
        if n_danger > 0:
            st.markdown(f"""
            <div class="alert-banner alert-danger">
                ⚠️ <strong>즉시 점검 필요 엔진 {n_danger}대</strong> — 잔여 수명 30사이클 미만 엔진이 있습니다. 즉시 확인하세요.
            </div>""", unsafe_allow_html=True)
        if n_warning > 0:
            st.markdown(f"""
            <div class="alert-banner alert-warning">
                🔔 <strong>점검 예약 권고 {n_warning}대</strong> — 30사이클 이내 점검이 필요한 엔진이 있습니다.
            </div>""", unsafe_allow_html=True)
 
        # ── 요약 지표 ──
        st.markdown('<p class="section-header">오늘의 엔진 현황 요약</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체 엔진", f"{total}대")
        c2.metric("🔴 즉시 점검", f"{n_danger}대")
        c3.metric("🟡 점검 예약", f"{n_warning}대")
        c4.metric("🟢 정상 운행", f"{n_normal}대")
 
        # ── 엔진 상태 카드 리스트 ──
        st.markdown('<p class="section-header">엔진별 상태 카드</p>', unsafe_allow_html=True)
 
        # 위험 → 주의 → 정상 순 정렬
        latest_sorted = latest.sort_values('pred_rul').reset_index(drop=True)
 
        cols_per_row = 3
        for i in range(0, len(latest_sorted), cols_per_row):
            row_data = latest_sorted.iloc[i:i+cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (_, row) in zip(cols, row_data.iterrows()):
                rul = int(row['pred_rul'])
                status_name, css_class, color, icon, action = get_rul_status(rul)
                max_rul = int(latest['pred_rul'].max()) or 1
                pct = min(100, int(rul / max_rul * 100))
                period = rul_to_period(rul)
 
                with col:
                    st.markdown(f"""
                    <div class="status-card {css_class}">
                        <div class="card-header">
                            <span class="card-engine-id">{icon} 엔진 #{int(row['unit_nr'])}</span>
                            <span class="badge {css_class}">{status_name}</span>
                        </div>
                        <div class="rul-value">{rul} <span style="font-size:0.85rem;color:#8b949e;">사이클</span></div>
                        <div class="rul-label">예상 잔여 수명 · {period}</div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width:{pct}%; background:{color};"></div>
                        </div>
                        <div class="action-text">→ {action}</div>
                    </div>
                    """, unsafe_allow_html=True)
 
        # ── 전체 분포 차트 ──
        st.markdown('<p class="section-header">잔여 수명 분포</p>', unsafe_allow_html=True)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=latest['pred_rul'],
            nbinsx=20,
            marker_color='#58a6ff',
            opacity=0.8,
            name='엔진 수'
        ))
        fig_dist.add_vline(x=30, line_dash="dash", line_color="#f85149", annotation_text="위험 기준 (30)", annotation_position="top right")
        fig_dist.add_vline(x=60, line_dash="dash", line_color="#d29922", annotation_text="주의 기준 (60)", annotation_position="top right")
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="잔여 수명 (사이클)",
            yaxis_title="엔진 수",
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            font=dict(family="Noto Sans KR"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
 
    else:
        st.info("데이터를 불러올 수 없습니다. DB 및 모델 파일을 확인하세요.")
 
# =========================================================
# 탭 2: 엔진 상세
# =========================================================
with tab_engine:
    summary_df, best_model = load_all_summary_best(subset_choice)
 
    if summary_df is not None:
        latest_all = summary_df.drop_duplicates('unit_nr', keep='last')
        target_row = latest_all[latest_all['unit_nr'] == unit_id]
 
        if not target_row.empty:
            pred_rul   = int(target_row['pred_rul'].iloc[0])
            # actual_rul은 내부 검증용이므로 UI에 노출하지 않음
            status_name, css_class, color, icon, action = get_rul_status(pred_rul)
            period = rul_to_period(pred_rul)
 
            # ── 알림 배너 (개별 엔진) ──
            if css_class == "danger":
                st.markdown(f'<div class="alert-banner alert-danger">⚠️ 엔진 #{unit_id}는 즉시 점검이 필요합니다. 잔여 수명이 {pred_rul}사이클({period})밖에 남지 않았습니다.</div>', unsafe_allow_html=True)
            elif css_class == "warning":
                st.markdown(f'<div class="alert-banner alert-warning">🔔 엔진 #{unit_id} 점검을 조속히 예약하세요. 잔여 수명 {pred_rul}사이클({period}) 남았습니다.</div>', unsafe_allow_html=True)
 
            # ── 엔진 상태 헤더 ──
            st.markdown(f'<p class="section-header">엔진 #{unit_id} 현재 상태 · {subset_choice}</p>', unsafe_allow_html=True)
 
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                st.markdown(f"""
                <div style="background:#161b22;border:1px solid #21262d;border-left:4px solid {color};border-radius:8px;padding:16px 18px;">
                    <div style="color:#8b949e;font-size:0.75rem;margin-bottom:4px;">종합 상태</div>
                    <div style="font-size:1.6rem;font-weight:700;color:{color};">{icon} {status_name}</div>
                </div>""", unsafe_allow_html=True)
            with h2:
                st.metric("예상 잔여 수명", f"{pred_rul} 사이클", help=f"약 {period}")
            with h3:
                st.metric("잔여 기간 (추정)", period)
 
            # ── RUL 잔여 게이지 ──
            # 데이터셋 내 최대 pred_rul을 기준으로 현재 엔진의 잔여 비율 계산
            # (actual_rul은 테스트셋 정답값이므로 운영 환경에서 알 수 없어 사용 안 함)
            dataset_max_rul = int(latest_all['pred_rul'].max()) if not latest_all.empty else 150
            remain_pct = min(100, int(pred_rul / dataset_max_rul * 100))
 
            st.markdown('<p class="section-header">수명 소모 현황</p>', unsafe_allow_html=True)
            st.caption(f"💡 데이터셋 내 최대 잔여 수명({dataset_max_rul}사이클) 대비 현재 엔진의 잔여 비율")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=remain_pct,
                title={'text': "잔여 수명 (%)", 'font': {'color': '#8b949e', 'size': 14}},
                number={'suffix': "%", 'font': {'color': '#e6edf3', 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8b949e'},
                    'bar': {'color': color},
                    'bgcolor': '#21262d',
                    'bordercolor': '#30363d',
                    'steps': [
                        {'range': [0, 20],  'color': 'rgba(248,81,73,0.15)'},
                        {'range': [20, 40], 'color': 'rgba(210,153,34,0.08)'},
                        {'range': [40, 100],'color': 'rgba(63,185,80,0.05)'},
                    ],
                    'threshold': {
                        'line': {'color': '#f85149', 'width': 2},
                        'thickness': 0.75, 'value': 20
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color='#8b949e', family='Noto Sans KR'),
                height=260,
                margin=dict(l=30, r=30, t=30, b=10),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
 
            # ── 부품별 위험도 ──
            st.markdown('<p class="section-header">부품별 이상 징후 추정</p>', unsafe_allow_html=True)
            df_raw = load_sensor_data(f"test_{subset_choice.lower()}", unit_id)
            useful_cols = [f"s_{i}" for i in USEFUL_SENSORS.get(subset_choice, [])]

            # 엔진 상세 탭: 클러스터 시리즈 준비 (FD002/004 전용)
            df_prep_tab2 = load_sensor_data(f"test_{subset_choice.lower()}_prep", unit_id)
            cluster_col_tab2 = None
            if subset_choice in ["FD002", "FD004"] and not df_prep_tab2.empty and 'op_cluster' in df_prep_tab2.columns:
                cl_merged = df_raw[['time_cycles']].merge(
                    df_prep_tab2[['time_cycles', 'op_cluster']], on='time_cycles', how='left'
                )
                cluster_col_tab2 = cl_merged['op_cluster']

            if not df_raw.empty and useful_cols:
                # 전체 데이터셋 기준선 확보 (클러스터 인식 버전, 캐시 활용)
                @st.cache_data
                def get_dataset_baselines(subset: str, useful: list):
                    """데이터셋 전체 엔진의 센서별 이상도 분포를 계산해 기준선 반환"""
                    try:
                        con = duckdb.connect('cmapss.db', read_only=True)
                        df_all = con.execute(
                            f"SELECT * FROM test_{subset.lower()} ORDER BY unit_nr, time_cycles"
                        ).df()
                        if subset in ["FD002", "FD004"]:
                            try:
                                df_prep_all = con.execute(
                                    f"SELECT unit_nr, time_cycles, op_cluster FROM test_{subset.lower()}_prep"
                                ).df()
                                df_all = df_all.merge(df_prep_all, on=['unit_nr', 'time_cycles'], how='left')
                            except:
                                pass
                        con.close()
                    except:
                        return {}, 40.0, 15.0
                    all_scores = []
                    for uid in df_all['unit_nr'].unique():
                        sub_df = df_all[df_all['unit_nr'] == uid]
                        cl_col = sub_df['op_cluster'] if 'op_cluster' in sub_df.columns else None
                        for s in useful:
                            if s in sub_df.columns and len(sub_df[s]) >= 3:
                                _, _, sc = detect_sensor_status(sub_df[s], window=10, cluster_series=cl_col)
                                # 기준선 계산 시 RUL 보정 미적용 (전체 엔진 평균이므로 중립)
                                all_scores.append(sc)
                    if not all_scores:
                        return {}, 40.0, 15.0
                    return {}, float(np.mean(all_scores)), float(np.std(all_scores))

                _, ds_mean, ds_std = get_dataset_baselines(subset_choice, useful_cols)

                part_cols = st.columns(len(PART_MAP))
                for col, (part_name, sensors) in zip(part_cols, PART_MAP.items()):
                    avail = [s for s in sensors if s in df_raw.columns and s in useful_cols]
                    if avail:
                        engine_max = max(
                            (detect_sensor_status(
                                df_raw[s], window=10,
                                cluster_series=cluster_col_tab2,
                                pred_rul=pred_rul,
                                dataset_max_rul=dataset_max_rul,
                            )[2] for s in avail),
                            default=0
                        )
                        # Z-score 기반 상대 위험도 → 0~100
                        z = (engine_max - ds_mean) / (ds_std + 1e-9)
                        risk_pct = int(min(100, max(0, 50 + z * 15)))

                        if risk_pct > 65:   r_color, r_label = "#f85149", "주의"
                        elif risk_pct > 50: r_color, r_label = "#d29922", "관찰"
                        else:               r_color, r_label = "#3fb950", "정상"

                        with col:
                            st.markdown(f"""
                            <div class="info-card" style="text-align:center;padding:14px 10px;">
                                <div style="color:#8b949e;font-size:0.75rem;margin-bottom:6px;">{part_name}</div>
                                <div style="font-size:1.4rem;font-weight:700;color:{r_color};">{risk_pct}%</div>
                                <div style="font-size:0.75rem;color:{r_color};margin-bottom:8px;">{r_label}</div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill" style="width:{risk_pct}%;background:{r_color};"></div>
                                </div>
                            </div>""", unsafe_allow_html=True)
 
            # ── 전수 엔진 RUL 비교 차트 ──
            st.markdown('<p class="section-header">전체 엔진 잔여 수명 비교</p>', unsafe_allow_html=True)
            comp = latest_all.sort_values('unit_nr')
            bar_colors = ['#f85149' if r < 30 else '#d29922' if r < 60 else '#3fb950'
                          for r in comp['pred_rul']]
            fig_bar = go.Figure(go.Bar(
                x=comp['unit_nr'],
                y=comp['pred_rul'],
                marker_color=bar_colors,
                name="예상 잔여 수명",
            ))
            # 현재 선택 엔진 강조
            fig_bar.add_vline(x=unit_id, line_dash="dash", line_color="#58a6ff",
                              annotation_text=f"현재: #{unit_id}", annotation_position="top")
            fig_bar.add_hline(y=30, line_dash="dot", line_color="#f85149", annotation_text="위험 기준")
            fig_bar.add_hline(y=60, line_dash="dot", line_color="#d29922", annotation_text="주의 기준")
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="엔진 번호",
                yaxis_title="잔여 수명 (사이클)",
                height=320,
                margin=dict(l=0, r=0, t=20, b=0),
                font=dict(family="Noto Sans KR"),
                showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
 
        else:
            st.warning(f"엔진 #{unit_id} 데이터가 없습니다.")
    else:
        st.info("데이터를 불러올 수 없습니다.")
 
# =========================================================
# 탭 3: 센서 분석
# =========================================================
with tab_sensor:
    st.markdown(f'<p class="section-header">엔진 #{unit_id} · 실시간 부품 모니터링 · {subset_choice}</p>', unsafe_allow_html=True)
 
    try:
        df_raw  = load_sensor_data(f"test_{subset_choice.lower()}", unit_id)
        df_prep = load_sensor_data(f"test_{subset_choice.lower()}_prep", unit_id)
        useful_cols = [f"s_{i}" for i in USEFUL_SENSORS.get(subset_choice, [])]
 
        if df_raw.empty:
            st.warning("센서 데이터가 없습니다.")
        else:
            # ── 클러스터 시리즈 준비 (FD002/004만 적용) ──
            cluster_col = None
            if subset_choice in ["FD002", "FD004"] and not df_prep.empty and 'op_cluster' in df_prep.columns:
                cluster_merged = df_raw[['time_cycles']].merge(
                    df_prep[['time_cycles', 'op_cluster']], on='time_cycles', how='left'
                )
                cluster_col = cluster_merged['op_cluster']

            # ── RUL 정보 준비 (이상 감지 보정용) ──
            _sum_tab3, _ = load_all_summary_best(subset_choice)
            _tab3_pred_rul, _tab3_max_rul = None, None
            if _sum_tab3 is not None:
                _latest_tab3 = _sum_tab3.drop_duplicates('unit_nr', keep='last')
                _target_tab3 = _latest_tab3[_latest_tab3['unit_nr'] == unit_id]
                if not _target_tab3.empty:
                    _tab3_pred_rul = int(_target_tab3['pred_rul'].iloc[0])
                _tab3_max_rul = int(_latest_tab3['pred_rul'].max()) if not _latest_tab3.empty else 150

            # 1. 상단 레이아웃 (엔진 진단 & 종합 상태)
            col_diag, col_info = st.columns([2, 1])

            # 부위별 대표 센서 매핑 (통계 기반 상태 결정)
            sensor_mapping = {'LPC': 's_2', 'HPC': 's_3', 'LPT': 's_15', 'Fan': 's_13', 'Combustor': 's_4'}
            current_status = {}
            worst_score = 0

            for comp, s_id in sensor_mapping.items():
                if s_id in df_raw.columns:
                    color, label, score = detect_sensor_status(
                        df_raw[s_id], window=10,
                        cluster_series=cluster_col,
                        pred_rul=_tab3_pred_rul,
                        dataset_max_rul=_tab3_max_rul,
                    )
                    current_status[comp] = color
                    worst_score = max(worst_score, score)
                else:
                    current_status[comp] = "#8b949e"

            with col_diag:
                st.plotly_chart(draw_engine_monitor(current_status), use_container_width=True)
            
            with col_info:
                # 모델 예측 대신 '통계적 위험 점수' 기반 종합 진단 표시
                status_name, s_color, s_msg = "정상", "#3fb950", "모든 센서가 안정 범위 내에 있습니다."
                if worst_score > 55:
                    status_name, s_color, s_msg = "위험", "#f85149", "주요 센서에서 급격한 이상 징후가 발견되었습니다. 즉각 점검이 필요합니다."
                elif worst_score > 30:
                    status_name, s_color, s_msg = "관찰", "#d29922", "일부 센서 수치가 평소보다 높게 관측되었습니다. 추세를 모니터링하세요."

                st.markdown(f"""
                <div class="info-card" style="text-align:center; padding:20px; border-top: 4px solid {s_color};">
                    <div style="font-size:0.9rem; color:#8b949e;">엔진 종합 상태 점수</div>
                    <div style="font-size:2.8rem; font-weight:bold; color:{s_color}; margin:10px 0;">
                        {int(100 - worst_score)} <span style="font-size:1.2rem;">/ 100</span>
                    </div>
                    <div style="padding:5px 15px; border-radius:20px; background:{s_color}22; color:{s_color}; display:inline-block; font-weight:bold;">
                        {status_name}
                    </div>
                    <div style="font-size:0.85rem; color:#e6edf3; margin-top:15px; line-height:1.4;">{s_msg}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── 현재 센서 지표 (Gauge Cards) ──
            st.markdown('<p class="section-header">현재 주요 센서 지표 (최신 사이클)</p>', unsafe_allow_html=True)
            last_row = df_raw.iloc[-1]
            avail_useful = [s for s in useful_cols if s in df_raw.columns]
            
            g_cols = st.columns(4)
            for idx, s in enumerate(avail_useful[:8]):
                meta = SENSOR_META.get(s, {"name": s, "unit": ""})
                val = last_row[s]
                color, status, score = detect_sensor_status(
                    df_raw[s], window=10,
                    cluster_series=cluster_col,
                    pred_rul=_tab3_pred_rul,
                    dataset_max_rul=_tab3_max_rul,
                )
                with g_cols[idx % 4]:
                    st.markdown(f"""
                    <div class="info-card" style="margin-bottom:10px;">
                        <div style="color:#8b949e;font-size:0.72rem;">{meta['name']}</div>
                        <div style="font-family:'JetBrains Mono';font-size:1.1rem;color:#e6edf3;">{val:.2f} <small>{meta['unit']}</small></div>
                        <div style="font-size:0.7rem;color:{color};">{status}</div>
                        <div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{min(score,100)}%;background:{color};"></div></div>
                    </div>""", unsafe_allow_html=True)

            # ── 센서 추세 분석 (Raw vs Normalized) ──
            st.markdown('<p class="section-header">센서 추세 분석</p>', unsafe_allow_html=True)
            all_s = sorted([c for c in df_raw.columns if re.match(r'^s_\d+$', c)], key=lambda x: int(x.split('_')[1]))
            selected_s = st.multiselect("분석 센서", options=all_s, default=avail_useful[:3], format_func=lambda x: sensor_label(x))

            if selected_s:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**실시간 센서 측정값**")
                    df_p = df_raw[['time_cycles'] + selected_s].copy()
                    df_p.rename(columns={s: sensor_label(s) for s in selected_s}, inplace=True)
                    df_p = df_p.melt('time_cycles', var_name='센서', value_name='측정값')
                    fig = px.line(df_p, x='time_cycles', y='측정값', color='센서',
                                  template="plotly_dark")
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="운전 사이클 수", yaxis_title="센서 측정값",
                        height=320, margin=dict(l=0,r=0,t=10,b=0),
                        font=dict(family="Noto Sans KR"),
                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.markdown("**운항 조건 보정 후 추세**")
                    if not df_prep.empty:
                        avail_p = [s for s in selected_s if s in df_prep.columns]
                        if avail_p:
                            df_pp = df_prep[['time_cycles'] + avail_p].copy()
                            df_pp.rename(columns={s: sensor_label(s) for s in avail_p}, inplace=True)
                            df_pp = df_pp.melt('time_cycles', var_name='센서', value_name='보정값')
                            fig_p = px.line(df_pp, x='time_cycles', y='보정값', color='센서',
                                            template="plotly_dark")
                            fig_p.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis_title="운전 사이클 수", yaxis_title="운항조건 보정값",
                                height=320, margin=dict(l=0,r=0,t=10,b=0),
                                font=dict(family="Noto Sans KR"),
                                legend=dict(bgcolor="rgba(0,0,0,0)"),
                            )
                            st.plotly_chart(fig_p, use_container_width=True)

            # ── RUL 연관성 분석 (Top 10) ──
            st.markdown('<p class="section-header">수명 감소와 연관된 핵심 센서 TOP 10</p>', unsafe_allow_html=True)
            st.caption("연관도가 높을수록 엔진 수명과 밀접하게 움직이는 센서입니다. 집중 모니터링 대상으로 활용하세요.")

            if not df_prep.empty and 'true_rul' in df_raw.columns:
                df_corr_data = df_prep.merge(df_raw[['time_cycles', 'true_rul']], on='time_cycles')
                s_cols = [c for c in df_prep.columns if re.match(r'^s_\d+$', c)]
                raw_corr = df_corr_data[s_cols + ['true_rul']].corr()['true_rul'].drop('true_rul').dropna()

                # 절댓값 기준 TOP 10 (방향 유지)
                top10 = raw_corr.reindex(raw_corr.abs().sort_values(ascending=False).index).head(10)

                # 차트용 데이터프레임 구성
                chart_rows = []
                for sensor, corr_val in top10.items():
                    abs_val = abs(corr_val)
                    korean_name = sensor_label(sensor)
                    direction = "수명 줄면 상승 📈" if corr_val < 0 else "수명 줄면 하강 📉"
                    if abs_val >= 0.7:   strength_color = "#f85149"
                    elif abs_val >= 0.4: strength_color = "#d29922"
                    else:                strength_color = "#3fb950"
                    chart_rows.append({
                        "센서": korean_name,
                        "연관도": abs_val,
                        "방향": direction,
                        "color": strength_color,
                    })
                chart_df = pd.DataFrame(chart_rows).sort_values("연관도")

                fig_top10 = go.Figure()
                fig_top10.add_trace(go.Bar(
                    x=chart_df["연관도"],
                    y=chart_df["센서"],
                    orientation='h',
                    marker_color=chart_df["color"].tolist(),
                    text=[f"{v:.2f}  {d}" for v, d in zip(chart_df["연관도"], chart_df["방향"])],
                    textposition="outside",
                    textfont=dict(color="#c9d1d9", size=11),
                    hovertemplate="<b>%{y}</b><br>연관도: %{x:.2f}<extra></extra>",
                ))
                fig_top10.add_vline(x=0.7, line_dash="dot", line_color="#f85149",
                                    annotation_text="강한 연관 (0.7)", annotation_position="top",
                                    annotation_font_color="#f85149")
                fig_top10.add_vline(x=0.4, line_dash="dot", line_color="#d29922",
                                    annotation_text="보통 연관 (0.4)", annotation_position="top",
                                    annotation_font_color="#d29922")
                fig_top10.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="수명 연관도 (0 = 무관, 1 = 매우 강한 연관)",
                    xaxis_range=[0, 1.15],
                    height=380,
                    margin=dict(l=0, r=120, t=20, b=0),
                    font=dict(family="Noto Sans KR", color="#c9d1d9"),
                    showlegend=False,
                )
                st.plotly_chart(fig_top10, use_container_width=True)

                # 1위 센서 강조 안내
                top1_sensor  = chart_df.iloc[-1]["센서"]
                top1_dir     = chart_df.iloc[-1]["방향"]
                top1_val     = chart_df.iloc[-1]["연관도"]
                st.markdown(f"""
                <div class="info-card" style="border-left:4px solid #58a6ff;padding:14px 18px;">
                    <span style="color:#58a6ff;font-weight:600;">💡 핵심 모니터링 센서</span><br>
                    <span style="color:#e6edf3;">현재 엔진에서 수명과 가장 강하게 연관된 센서는
                    <strong>{top1_sensor}</strong>입니다. (연관도 {top1_val:.2f})</span><br>
                    <span style="color:#8b949e;font-size:0.82rem;">이 센서가 {top1_dir.split()[0]} 추세를 보일 경우 즉시 점검을 고려하세요.</span>
                </div>""", unsafe_allow_html=True)


            # ── SHAP 기반 RUL 예측 설명 ──
            st.markdown('<p class="section-header">AI가 설명하는 수명 예측 근거 (SHAP)</p>', unsafe_allow_html=True)
            st.caption("이 엔진의 잔여 수명을 낮게/높게 만든 센서가 무엇인지 AI가 설명합니다.")

            @st.cache_data
            def compute_shap(subset: str, unit_nr: int):
                """SHAP 값 계산 — 모델 파일 자동 탐색"""
                best_model_map = {
                    "FD001": "LightGBM", "FD002": "XGBoost",
                    "FD003": "XGBoost",  "FD004": "XGBoost",
                }
                model_name = best_model_map.get(subset, "XGBoost")
                model_path = f"saved_models/model_{model_name}_{subset}.pkl"

                if not os.path.exists(model_path):
                    return None, None, None

                # features 파일은 없을 수 있으므로 DB prep 컬럼에서 직접 추출
                feat_path = f"saved_models/features_{model_name}_{subset}.pkl"

                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)

                    con = duckdb.connect('cmapss.db', read_only=True)
                    df_input = con.execute(
                        f"SELECT * FROM test_{subset.lower()}_prep WHERE unit_nr = {unit_nr} ORDER BY time_cycles DESC LIMIT 1"
                    ).df()
                    con.close()

                    if df_input.empty:
                        return None, None, None

                    # 피처명 우선순위:
                    # 1) features_*.pkl 별도 파일
                    # 2) model.feature_names_in_ (sklearn 호환)
                    # 3) model.feature_name_ (LightGBM)
                    # 4) prep 테이블의 센서 컬럼 전체
                    if os.path.exists(feat_path):
                        with open(feat_path, 'rb') as f:
                            feature_names = pickle.load(f)
                    elif hasattr(model, 'feature_names_in_'):
                        feature_names = list(model.feature_names_in_)
                    elif hasattr(model, 'feature_name_'):
                        feature_names = list(model.feature_name_())
                    else:
                        # prep 테이블에서 센서/운전조건 컬럼만 추출
                        feature_names = [c for c in df_input.columns
                                         if c not in ('unit_nr', 'time_cycles', 'true_rul',
                                                       'op_cluster', 'op_setting_1',
                                                       'op_setting_2', 'op_setting_3')]

                    avail_feats = [f for f in feature_names if f in df_input.columns]
                    if not avail_feats:
                        return None, None, None

                    X = df_input[avail_feats].values

                    if hasattr(model, 'predict'):
                        explainer = shap.TreeExplainer(model)
                        shap_vals = explainer.shap_values(X)
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[0]
                        base = explainer.expected_value
                        if isinstance(base, (list, np.ndarray)):
                            base = float(base[0])
                        return shap_vals[0], avail_feats, float(base)
                except Exception:
                    return None, None, None

            if SHAP_AVAILABLE:
                shap_vals, feat_names, base_val = compute_shap(subset_choice, unit_id)

                if shap_vals is not None and feat_names:
                    # SHAP 값을 센서 한글 명칭으로 매핑해 정렬
                    shap_df = pd.DataFrame({
                        "sensor_code": feat_names,
                        "shap_value":  shap_vals,
                        "abs_shap":    np.abs(shap_vals),
                    })
                    shap_df["sensor_name"] = shap_df["sensor_code"].apply(
                        lambda s: SENSOR_META[s]["name"] + f" ({SENSOR_META[s]['symbol']})"
                              if s in SENSOR_META else s
                    )
                    shap_top = shap_df.nlargest(10, "abs_shap").sort_values("abs_shap")

                    # 색상: 양수(수명 증가 기여) → 파랑, 음수(수명 감소 기여) → 빨강
                    bar_colors = ["#58a6ff" if v > 0 else "#f85149" for v in shap_top["shap_value"]]

                    fig_shap = go.Figure()
                    fig_shap.add_trace(go.Bar(
                        x=shap_top["shap_value"],
                        y=shap_top["sensor_name"],
                        orientation='h',
                        marker_color=bar_colors,
                        text=[f"{'+' if v>0 else ''}{v:.1f}" for v in shap_top["shap_value"]],
                        textposition="outside",
                        textfont=dict(color="#c9d1d9", size=11),
                        hovertemplate="<b>%{y}</b><br>영향도: %{x:.2f} 사이클<extra></extra>",
                    ))
                    fig_shap.add_vline(x=0, line_color="#30363d", line_width=1)
                    fig_shap.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title="수명 예측 영향도 (사이클 단위 / 오른쪽=수명 증가, 왼쪽=수명 감소)",
                        height=380,
                        margin=dict(l=0, r=80, t=10, b=0),
                        font=dict(family="Noto Sans KR", color="#c9d1d9"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)

                    # 상위 3개 자연어 설명 자동 생성
                    top3_neg = shap_top[shap_top["shap_value"] < 0].nsmallest(3, "shap_value")
                    top3_pos = shap_top[shap_top["shap_value"] > 0].nlargest(3, "shap_value")

                    explain_lines = []
                    for _, row in top3_neg.iterrows():
                        explain_lines.append(
                            f"🔴 <strong>{row['sensor_name']}</strong>이(가) 현재 수치가 높아 "
                            f"예상 수명을 <strong>{abs(row['shap_value']):.1f}사이클 단축</strong>시키고 있습니다."
                        )
                    for _, row in top3_pos.iterrows():
                        explain_lines.append(
                            f"🔵 <strong>{row['sensor_name']}</strong>이(가) 안정적으로 유지되어 "
                            f"예상 수명에 <strong>{row['shap_value']:.1f}사이클 기여</strong>하고 있습니다."
                        )

                    if explain_lines:
                        explain_html = "<br>".join(explain_lines)
                        st.markdown(f"""
                        <div class="info-card" style="border-left:4px solid #58a6ff;padding:16px 20px;line-height:2;">
                            <div style="color:#58a6ff;font-weight:600;margin-bottom:10px;">🤖 AI 수명 예측 근거 요약</div>
                            <div style="color:#c9d1d9;font-size:0.88rem;">{explain_html}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("💡 SHAP 분석을 위해 saved_models/model_[모델명]_[데이터셋].pkl 파일이 필요합니다. (예: model_XGBoost_FD001.pkl)")
            else:
                st.warning("SHAP 라이브러리가 설치되지 않았습니다. `pip install shap` 후 재시작하세요.")

    except Exception as e:
        st.error(f"데이터 로드 및 분석 중 오류: {e}")

# =========================================================
# 탭 4: 점검 이력
# =========================================================
with tab_history:
    st.markdown(f'<p class="section-header">엔진 #{unit_id} 점검 이력</p>', unsafe_allow_html=True)
 
    if "history_records" not in st.session_state:
        st.session_state.history_records = {unit_id: MOCK_HISTORY.copy()}
    if unit_id not in st.session_state.history_records:
        st.session_state.history_records[unit_id] = []
 
    records = st.session_state.history_records[unit_id]

    # ── 점검 이력 요약 통계 ──
    if records:
        total_cnt   = len(records)
        last_date   = records[0]['date']
        urgent_cnt  = sum(1 for r in records if r['type'] == '긴급점검')
        regular_cnt = sum(1 for r in records if r['type'] == '정기점검')

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("총 점검 횟수",   f"{total_cnt}회")
        s2.metric("마지막 점검일",  last_date)
        s3.metric("정기점검 횟수",  f"{regular_cnt}회")
        s4.metric("🚨 긴급점검 횟수", f"{urgent_cnt}회")
        st.markdown("---")

    # ── 이력 목록 ──
    type_css = {"정기점검": "type-regular", "긴급점검": "type-urgent", "일상점검": "type-check"}
 
    if records:
        rows_html = ""
        for r in records:
            badge_cls = type_css.get(r['type'], 'type-check')
            rows_html += f"""
            <div class="history-row">
                <span class="history-date">{r['date']}</span>
                <span class="history-type-badge {badge_cls}">{r['type']}</span>
                <span>{r['note']}</span>
                <span style="color:#8b949e;font-size:0.8rem;">{r['by']}</span>
            </div>"""
        st.markdown(f'<div class="info-card">{rows_html}</div>', unsafe_allow_html=True)
    else:
        st.info("등록된 점검 이력이 없습니다.")
 
    # ── 신규 기록 추가 ──
    st.markdown('<p class="section-header">점검 기록 추가</p>', unsafe_allow_html=True)
    with st.form("add_history_form", clear_on_submit=True):
        fc1, fc2, fc3 = st.columns([1.2, 1.2, 2])
        with fc1:
            h_type = st.selectbox("점검 유형", ["일상점검", "정기점검", "긴급점검"])
        with fc2:
            h_by = st.text_input("담당자", placeholder="홍길동")
        with fc3:
            h_note = st.text_input("점검 내용", placeholder="예: 팬 블레이드 교체 완료")
 
        submitted = st.form_submit_button("✅ 기록 저장", use_container_width=True)
        if submitted and h_note:
            new_rec = {
                "date": datetime.now().strftime("%Y.%m.%d"),
                "type": h_type,
                "note": h_note,
                "by": h_by or "미입력",
            }
            st.session_state.history_records[unit_id].insert(0, new_rec)
            st.success("점검 기록이 저장되었습니다.")
            st.rerun()
 
    # ── AI 이력 기반 분석 ──
    st.markdown('<p class="section-header">AI 점검 이력 분석</p>', unsafe_allow_html=True)
    if st.button("🤖 이 엔진의 점검 패턴 분석해줘", use_container_width=True):
        history_text = "\n".join([f"{r['date']} {r['type']}: {r['note']} (담당:{r['by']})" for r in records])
        ai_agent = load_ai_agent()
        if ai_agent:
            with st.spinner("AI가 이력을 분석하고 있습니다..."):
                query = f"엔진 #{unit_id}의 아래 점검 이력을 바탕으로 반복되는 문제 패턴과 향후 주의사항을 정비사가 이해하기 쉽게 설명해주세요.\n\n{history_text}"
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    resp = loop.run_until_complete(run_chat(ai_agent, subset_choice, unit_id, query))
                    loop.close()
                    st.markdown(f'<div class="info-card"><p style="color:#c9d1d9;line-height:1.7;">{resp}</p></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"오류: {e}")

# =========================================================
# 탭 5: 운영 시나리오 — 헬퍼 함수
# =========================================================

@st.cache_data
def get_engine_sensor_diagnosis(subset: str, uid: int, useful_cols: list) -> dict:
    """
    특정 엔진의 마지막 사이클 센서를 분석해
    이상 센서 목록과 부품별 원인 추정을 반환.

    반환 예:
    {
      "anomalies": [
          {"sensor": "HPC 출구 온도 (T30)", "direction": "상승", "severity": "high", "part": "HPC 압축기"},
          ...
      ],
      "dominant_part": "HPC 압축기",
      "summary": "HPC 출구 온도와 압력이 기준보다 높게 유지되고 있어 ..."
    }
    """
    try:
        con = duckdb.connect('cmapss.db', read_only=True)
        # 해당 엔진 전체 + 마지막 행
        df = con.execute(
            f"SELECT * FROM test_{subset.lower()} WHERE unit_nr = {uid} ORDER BY time_cycles"
        ).df()
        con.close()
    except:
        return {}

    if df.empty or len(df) < 5:
        return {}

    last = df.iloc[-1]
    window = min(15, len(df))
    recent = df.tail(window)

    anomalies = []
    part_score = {}  # 부품별 이상 누적 점수

    # 부품-센서 매핑 (역방향)
    sensor_to_part = {}
    for part, sensors in PART_MAP.items():
        for s in sensors:
            sensor_to_part[s] = part

    for col in useful_cols:
        if col not in df.columns:
            continue
        arr = df[col].values
        global_range = arr.max() - arr.min()
        # 상수 센서 제외
        if global_range < abs(arr.mean()) * 0.001 + 1e-9:
            continue

        # 최근 기울기
        x = np.arange(window)
        recent_arr = recent[col].values
        slope = np.polyfit(x, recent_arr, 1)[0]
        global_std = arr.std() + 1e-9
        norm_slope = abs(slope) / global_std

        # 현재값 백분위
        pct = float((arr < last[col]).mean())
        extremity = abs(pct - 0.5) * 2

        score = min(1.0, norm_slope / 0.5) * 0.6 + extremity * 0.4
        if score < 0.30:
            continue  # 정상 센서 제외

        direction = "상승" if slope > 0 else "하강"
        severity  = "high" if score > 0.55 else "medium"
        part      = sensor_to_part.get(col, "기타")
        meta      = SENSOR_META.get(col, {"name": col, "unit": ""})
        sensor_name = f"{meta['name']} ({meta.get('symbol', col)})"

        anomalies.append({
            "sensor":    sensor_name,
            "code":      col,
            "direction": direction,
            "severity":  severity,
            "score":     round(score, 2),
            "part":      part,
            "pct_rank":  round(pct * 100, 1),
        })
        part_score[part] = part_score.get(part, 0) + score

    # 가장 이상 점수 높은 부품
    dominant_part = max(part_score, key=part_score.get) if part_score else None

    # 자연어 요약 생성
    if anomalies:
        top = sorted(anomalies, key=lambda x: x["score"], reverse=True)[:3]
        summary_parts = []
        for a in top:
            rank_txt = "매우 높은" if a["pct_rank"] > 80 else "높은" if a["pct_rank"] > 65 else "낮은"
            summary_parts.append(
                f"{a['sensor']}이(가) {rank_txt} 수준으로 {a['direction']}하고 있음"
            )
        summary = " · ".join(summary_parts)
    else:
        summary = "센서 수치는 안정적이나 누적 운전 사이클로 인해 잔여 수명이 임박했습니다."

    return {
        "anomalies":      anomalies,
        "dominant_part":  dominant_part,
        "summary":        summary,
        "part_score":     part_score,
    }


def render_scenario_card(uid: int, rul: int, days: str, diag: dict,
                          border_color: str, badge_color: str,
                          badge_bg: str, badge_text: str, urgency: str):
    """엔진별 개인화된 시나리오 카드 렌더링 — HTML을 변수로 조립 후 단일 markdown 호출"""
    anomalies = diag.get("anomalies", [])
    dom_part  = diag.get("dominant_part")
    summary   = diag.get("summary", "")

    # ── 이상 센서 뱃지 ──
    badges_html = ""
    for a in anomalies[:4]:
        c    = "#f85149" if a["severity"] == "high" else "#d29922"
        icon = "📈" if a["direction"] == "상승" else "📉"
        badges_html += (
            '<span style="display:inline-block;background:rgba(255,255,255,0.06);'
            f'border:1px solid {c};color:{c};font-size:0.72rem;'
            f'padding:2px 8px;border-radius:10px;margin:2px;">'
            f'{icon} {a["sensor"]} {a["direction"]}</span>'
        )

    # ── 부품별 권고 액션 ──
    part_actions = {
        "HPC":            ["HPC 압축기 블레이드 육안 점검", "압축기 세정(Water Wash) 실시", "HPC 출구 온도·압력 센서 교정"],
        "팬 시스템":       ["팬 블레이드 균열·마모 점검", "팬 베어링 이상 소음 확인", "팬 입구 이물질(FOD) 검사"],
        "LPC":            ["LPC 블레이드 간극 측정", "LPC 출구 온도 이력 검토", "저압 압축기 시일 상태 확인"],
        "터빈 (LPT/HPT)": ["터빈 블레이드 열손상 점검", "냉각 블리드 유량 측정", "터빈 케이싱 크리프 검사"],
        "코어":            ["코어 속도 센서 교정", "엔진 압력비(EPR) 재측정", "연료 노즐 막힘 여부 확인"],
    }
    default_actions = ["전체 보어스코프 검사 실시", "주요 센서 교정 점검", "수석 정비사 확인 후 운항 재개 결정"]
    actions   = part_actions.get(dom_part, default_actions)
    nums      = ["①", "②", "③"]
    act_lines = "".join(
        f'<div style="margin-left:8px;">{nums[i]} {a}</div>'
        for i, a in enumerate(actions[:3])
    )

    # ── 주요 이상 부품 표시 ──
    part_span = ""
    if dom_part:
        part_span = (
            f'<span style="color:#8b949e;font-size:0.78rem;margin-left:8px;">'
            f'· 주요 이상 부품: <strong style="color:{border_color};">{dom_part}</strong></span>'
        )

    # ── 요약 문구 ──
    summary_block = ""
    if summary:
        summary_block = (
            f'<div style="color:#8b949e;font-size:0.80rem;margin:8px 0 6px 0;line-height:1.6;">'
            f'📊 {summary}</div>'
        )

    # ── 센서 이상 없이 수명만 임박한 경우 ──
    no_anomaly = ""
    if not anomalies:
        no_anomaly = (
            '<div style="color:#8b949e;font-size:0.80rem;margin-bottom:6px;">'
            '💡 센서 수치 자체는 안정적이나 누적 운전에 의한 예방 점검이 필요합니다.</div>'
        )

    # ── 센서 뱃지 래퍼 ──
    badges_block = f'<div style="margin-bottom:6px;">{badges_html}</div>' if badges_html else ""

    # ── 최종 HTML 조립 (f-string 중첩 없이 변수만 삽입) ──
    html = (
        f'<div class="info-card" style="border-left:4px solid {border_color};margin-bottom:12px;">'
        f'  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px;">'
        f'    <div>'
        f'      <span style="color:{border_color};font-weight:700;font-size:1rem;">엔진 #{uid}</span>'
        f'      <span style="color:#8b949e;font-size:0.82rem;margin-left:10px;">잔여 수명 {rul}사이클 ({days})</span>'
        f'      {part_span}'
        f'    </div>'
        f'    <span style="background:{badge_bg};color:{badge_color};font-size:0.72rem;'
        f'padding:3px 10px;border-radius:12px;font-weight:600;">{badge_text}</span>'
        f'  </div>'
        f'  {badges_block}'
        f'  {summary_block}'
        f'  {no_anomaly}'
        f'  <div style="font-size:0.83rem;color:#c9d1d9;line-height:1.9;margin-top:6px;">'
        f'    📋 <strong>권고 조치</strong>'
        f'    {act_lines}'
        f'  </div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# 탭 5: 운영 시나리오
# =========================================================
with tab_scenario:
    st.markdown('<p class="section-header">전체 엔진 운영 시나리오 자동 생성</p>', unsafe_allow_html=True)
    st.caption("AI가 잔여 수명 데이터를 분석해 즉시 실행 가능한 점검 시나리오와 일정을 자동으로 생성합니다.")

    summary_sc, _ = load_all_summary_best(subset_choice)

    if summary_sc is not None:
        latest_sc = summary_sc.drop_duplicates('unit_nr', keep='last').sort_values('pred_rul')

        # ── 긴급도별 분류 ──
        critical = latest_sc[latest_sc['pred_rul'] < 30]
        warning  = latest_sc[(latest_sc['pred_rul'] >= 30) & (latest_sc['pred_rul'] < 60)]
        normal   = latest_sc[latest_sc['pred_rul'] >= 60]

        # ── 상단 요약 지표 ──
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("총 엔진 수",       f"{len(latest_sc)}대")
        sc2.metric("🔴 즉시 점검 필요", f"{len(critical)}대")
        sc3.metric("🟡 이번 주 점검",   f"{len(warning[warning['pred_rul'] < 45])}대")
        sc4.metric("📅 이번 달 점검",   f"{len(warning[warning['pred_rul'] >= 45])}대")

        st.markdown("---")

        # ── 자동 생성 점검 시나리오 카드 ──
        st.markdown('<p class="section-header">🔴 즉시 점검 필요 엔진 — 운항 즉시 중단 권고</p>', unsafe_allow_html=True)
        if critical.empty:
            st.markdown('<div class="info-card"><span style="color:#3fb950;">✅ 즉시 점검이 필요한 엔진이 없습니다.</span></div>', unsafe_allow_html=True)
        else:
            useful_sc = [f"s_{i}" for i in USEFUL_SENSORS.get(subset_choice, [])]
            for _, row in critical.iterrows():
                uid  = int(row['unit_nr'])
                rul  = int(row['pred_rul'])
                days = rul_to_period(rul)
                diag = get_engine_sensor_diagnosis(subset_choice, uid, useful_sc)
                render_scenario_card(
                    uid, rul, days, diag,
                    border_color="#f85149",
                    badge_color="#f85149",
                    badge_bg="rgba(248,81,73,0.15)",
                    badge_text="즉시 조치",
                    urgency="critical",
                )

        st.markdown('<p class="section-header">🟡 이번 주 점검 예약 권고 (30~45사이클)</p>', unsafe_allow_html=True)
        week_warning = warning[warning['pred_rul'] < 45]
        if week_warning.empty:
            st.markdown('<div class="info-card"><span style="color:#3fb950;">✅ 이번 주 긴급 예약이 필요한 엔진이 없습니다.</span></div>', unsafe_allow_html=True)
        else:
            useful_sc = [f"s_{i}" for i in USEFUL_SENSORS.get(subset_choice, [])]
            for _, row in week_warning.iterrows():
                uid  = int(row['unit_nr'])
                rul  = int(row['pred_rul'])
                days = rul_to_period(rul)
                diag = get_engine_sensor_diagnosis(subset_choice, uid, useful_sc)
                render_scenario_card(
                    uid, rul, days, diag,
                    border_color="#d29922",
                    badge_color="#d29922",
                    badge_bg="rgba(210,153,34,0.15)",
                    badge_text="이번 주 예약",
                    urgency="warning",
                )

        st.markdown('<p class="section-header">📅 이번 달 점검 예약 권고 (45~60사이클)</p>', unsafe_allow_html=True)
        month_warning = warning[warning['pred_rul'] >= 45]
        if month_warning.empty:
            st.markdown('<div class="info-card"><span style="color:#3fb950;">✅ 이번 달 추가 예약이 필요한 엔진이 없습니다.</span></div>', unsafe_allow_html=True)
        else:
            # 많을 경우 최대 10개만 표시
            for _, row in month_warning.head(10).iterrows():
                uid  = int(row['unit_nr'])
                rul  = int(row['pred_rul'])
                days = rul_to_period(rul)
                st.markdown(f"""
                <div class="info-card" style="border-left:4px solid #58a6ff;margin-bottom:8px;padding:12px 18px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                            <span style="color:#58a6ff;font-weight:600;">📅 엔진 #{uid}</span>
                            <span style="color:#8b949e;font-size:0.82rem;margin-left:10px;">잔여 수명 {rul}사이클 ({days})</span>
                        </div>
                        <span style="background:rgba(88,166,255,0.12);color:#58a6ff;font-size:0.72rem;padding:3px 10px;border-radius:12px;">이번 달 예약</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            if len(month_warning) > 10:
                st.caption(f"외 {len(month_warning)-10}대 추가")

        # ── 점검 일정 캘린더 뷰 (Gantt 스타일) ──
        st.markdown("---")
        st.markdown('<p class="section-header">점검 일정 시각화 (예상 점검 필요 시점)</p>', unsafe_allow_html=True)
        st.caption("각 바는 현재 시점부터 해당 엔진의 점검 필요 시점까지를 나타냅니다.")

        gantt_data = latest_sc[latest_sc['pred_rul'] < 100].copy()
        if not gantt_data.empty:
            gantt_data = gantt_data.sort_values('pred_rul').head(30)
            gantt_colors = [
                "#f85149" if r < 30 else "#d29922" if r < 60 else "#3fb950"
                for r in gantt_data['pred_rul']
            ]
            fig_gantt = go.Figure()
            fig_gantt.add_trace(go.Bar(
                x=gantt_data['pred_rul'],
                y=[f"엔진 #{int(u)}" for u in gantt_data['unit_nr']],
                orientation='h',
                marker_color=gantt_colors,
                text=[f"{int(r)}사이클 ({rul_to_period(int(r))})" for r in gantt_data['pred_rul']],
                textposition='outside',
                textfont=dict(color='#c9d1d9', size=10),
            ))
            fig_gantt.add_vline(x=30, line_dash="dot", line_color="#f85149",
                                annotation_text="즉시 점검 기준", annotation_position="top",
                                annotation_font_color="#f85149")
            fig_gantt.add_vline(x=60, line_dash="dot", line_color="#d29922",
                                annotation_text="이번 달 점검 기준", annotation_position="top",
                                annotation_font_color="#d29922")
            fig_gantt.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="현재로부터 점검 필요까지 남은 사이클",
                height=max(350, len(gantt_data) * 24),
                margin=dict(l=0, r=100, t=20, b=0),
                font=dict(family="Noto Sans KR", color="#c9d1d9"),
                showlegend=False,
            )
            st.plotly_chart(fig_gantt, use_container_width=True)

        # ── 보고서 다운로드 ──
        st.markdown("---")
        st.markdown('<p class="section-header">📥 점검 시나리오 보고서 다운로드</p>', unsafe_allow_html=True)

        def generate_report(df_latest, subset_name):
            lines = [
                f"엔진 상태 관리 시스템 — 점검 시나리오 보고서",
                f"데이터셋: {subset_name}",
                f"생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}",
                "=" * 60,
                "",
                "[즉시 점검 필요 — 운항 중단 권고]",
            ]
            for _, row in df_latest[df_latest['pred_rul'] < 30].iterrows():
                lines.append(f"  • 엔진 #{int(row['unit_nr'])}: 잔여 {int(row['pred_rul'])}사이클 ({rul_to_period(int(row['pred_rul']))})")
            lines += ["", "[이번 주 점검 예약 권고 (30~45사이클)]"]
            for _, row in df_latest[(df_latest['pred_rul'] >= 30) & (df_latest['pred_rul'] < 45)].iterrows():
                lines.append(f"  • 엔진 #{int(row['unit_nr'])}: 잔여 {int(row['pred_rul'])}사이클 ({rul_to_period(int(row['pred_rul']))})")
            lines += ["", "[이번 달 점검 예약 권고 (45~60사이클)]"]
            for _, row in df_latest[(df_latest['pred_rul'] >= 45) & (df_latest['pred_rul'] < 60)].iterrows():
                lines.append(f"  • 엔진 #{int(row['unit_nr'])}: 잔여 {int(row['pred_rul'])}사이클 ({rul_to_period(int(row['pred_rul']))})")
            lines += ["", "=" * 60, "본 보고서는 AI 예측 기반이며 최종 판단은 수석 정비사가 확인하세요."]
            return "\n".join(lines)

        report_text = generate_report(latest_sc, subset_choice)
        st.download_button(
            label="📄 텍스트 보고서 다운로드 (.txt)",
            data=report_text.encode("utf-8"),
            file_name=f"점검시나리오_{subset_choice}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # CSV 다운로드
        csv_data = latest_sc[['unit_nr', 'pred_rul']].copy()
        csv_data['잔여기간'] = csv_data['pred_rul'].apply(lambda r: rul_to_period(int(r)))
        csv_data['점검긴급도'] = csv_data['pred_rul'].apply(
            lambda r: "즉시점검" if r < 30 else "이번주" if r < 45 else "이번달" if r < 60 else "정상"
        )
        csv_data.columns = ['엔진번호', '잔여수명(사이클)', '잔여기간', '점검긴급도']
        st.download_button(
            label="📊 엑셀용 CSV 다운로드 (.csv)",
            data=csv_data.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"엔진현황_{subset_choice}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("데이터를 불러올 수 없습니다.")
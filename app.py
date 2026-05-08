import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
import duckdb
import google.generativeai as genai
import re

# --- 1. 초기 설정 및 API 연결 ---
st.set_page_config(layout="wide", page_title="CMAPSS Analysis Dashboard")

# Gemini API 설정
GEMINI_API_KEY = "Key"
genai.configure(api_key=GEMINI_API_KEY)

# --- 모델 로드 함수 ---
@st.cache_resource
def load_gemini_model():
    try:
        model_list = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        target = next((m for m in ['models/gemini-1.5-flash', 'gemini-1.5-flash'] if m in model_list), None)
        if not target and model_list: target = model_list[0]
        return genai.GenerativeModel(target) if target else None
    except Exception: 
        return None

# --- 3. 데이터 로드 및 계산 함수 (순서 변경: run_chain_analysis에서 호출하므로 위로 이동) ---
def calculate_nasa_score(actual, pred):
    d = pred - actual
    score = np.where(d < 0, np.exp(-d/13)-1, np.exp(d/10)-1)
    return np.sum(score)

@st.cache_resource
def get_connection():
    return duckdb.connect('cmapss.db', read_only=True)

@st.cache_resource
def load_analysis_data(subset, model_name):
    base = "saved_models"
    history_path = f"{base}/history_{model_name}_{subset}.csv"
    summary_path = f"{base}/summary_{model_name}_{subset}.csv"
    asset_path = f"{base}/preprocess_{subset}.pkl" 
    data = {}
    try:
        if os.path.exists(history_path): data['history'] = pd.read_csv(history_path)
        if os.path.exists(summary_path):
            df = pd.read_csv(summary_path)
            if 'unit_nr' not in df.columns: df['unit_nr'] = df.index + 1
            data['summary'] = df
        if os.path.exists(asset_path): 
            with open(asset_path, 'rb') as f: data['assets'] = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None

# --- 2. DB 연동형 프롬프트 체이닝 챗봇 함수 (model_choice를 인자로 받도록 수정) ---
def run_chain_analysis(model, subset, unit_nr, user_query, current_model_choice):
    if model is None: return "❌ 모델 로드 실패"

    try:
        # 1. 질문 분석
        extract_prompt = f"질문: '{user_query}', 현재설정: {subset}, {unit_nr}. 결과만 'FD00X, 번호' 형식으로 답하세요."
        extracted = model.generate_content(extract_prompt).text.strip()
        match = re.search(r'(FD00\d)\D*(\d+)', extracted)
        target_subset, target_unit = (match.group(1), int(match.group(2))) if match else (subset, unit_nr)

        # 2. DB 연결
        con = duckdb.connect('cmapss.db', read_only=True)
        latest_df = con.execute(f"SELECT * FROM test_{target_subset.lower()} WHERE unit_nr = {target_unit} ORDER BY time_cycles DESC LIMIT 1").df()
        con.close()

        # 3. 예측값 찾기 (전달받은 current_model_choice 사용)
        analysis_data = load_analysis_data(target_subset, current_model_choice)
        pred_val = "데이터 없음"
        if analysis_data and 'summary' in analysis_data:
            summary_df = analysis_data['summary']
            target_row = summary_df[summary_df['unit_nr'] == target_unit]
            if not target_row.empty:
                pred_val = round(target_row['pred_rul'].iloc[-1], 2)

        if latest_df.empty:
            return f"❌ {target_subset} #{target_unit} 데이터를 DB에서 찾을 수 없습니다."

        engine_status = {
            "분석_대상": f"{target_subset} 엔진 #{target_unit}",
            "현재_사이클": int(latest_df['time_cycles'].iloc[0]),
            "모델_예측_RUL": pred_val,
            "실제_정답_RUL": int(latest_df['true_rul'].iloc[0]) if 'true_rul' in latest_df.columns else "알 수 없음",
            "센서_데이터": latest_df.to_dict(orient='records')[0]
        }

        final_prompt = f"""
        당신은 항공 엔진 전문가입니다. 아래 데이터를 근거로 답변하세요.
        {engine_status}
        
        [지침] 
        1. '모델_예측_RUL'과 '실제_정답_RUL'이 다를 경우, 모델이 실제보다 얼마나 높게/낮게 예측했는지 언급하세요.
        2. {target_subset} #{target_unit}에 대한 질문임을 명시하세요.
        3. 사용자가 '센서 X번'에 대해 물으면, 진단 보고서를 쓰기 전에 해당 센서의 이름(예: Nc, 코어 속도)부터 명확히 답하세요.
        4. '엔진 번호'와 '센서 번호'를 절대 혼동하지 마세요.
        5. 데이터셋={subset}, 엔진={unit_nr}
        위 질문에서 '데이터셋(FD001~4)'과 '엔진 번호(숫자)'를 각각 찾아내세요.
            - 결과는 반드시 'DATASET: FD00X, UNIT: 번호' 형식으로만 출력하세요.
        """
        return model.generate_content(final_prompt).text
    except Exception as e:
        return f"❌ 분석 중 오류: {str(e)}"

def load_sensor_data(table_name, unit_nr):
    con = get_connection()
    query = f"SELECT * FROM {table_name} WHERE unit_nr = {unit_nr} ORDER BY time_cycles"
    return con.execute(query).df()

# --- 4. 사이드바 UI 및 챗봇 ---
st.sidebar.title("🔍 설정 및 AI 챗봇")
subset_choice = st.sidebar.selectbox("데이터셋 선택", ["FD001", "FD002", "FD003", "FD004"])
model_choice = st.sidebar.selectbox("모델 선택", ["XGBoost", "LightGBM", "RF", "Ridge", "Lasso"])
unit_id_input = st.sidebar.number_input("엔진(Unit) 번호", min_value=1, value=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 데이터 분석가 (Chain)")
gemini_model = load_gemini_model()

if "messages" not in st.session_state: st.session_state.messages = []
chat_container = st.sidebar.container(height=350)
for msg in st.session_state.messages: chat_container.chat_message(msg["role"]).write(msg["content"])

if prompt := st.sidebar.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    chat_container.chat_message("user").write(prompt)
    with chat_container.chat_message("assistant"):
        with st.spinner("DB 분석 중..."):
            # 수정: model_choice를 명시적으로 전달
            resp = run_chain_analysis(gemini_model, subset_choice, unit_id_input, prompt, model_choice)
            st.write(resp)
    st.session_state.messages.append({"role": "assistant", "content": resp})

# --- 5. 메인 화면 ---
st.title(f"📊 CMAPSS 분석 대시보드 ({subset_choice})")

tabs = st.tabs(["🏠 홈 / 개요", "🎯 모델 성능 분석", "📈 센서 추세 분석"])
tab_home, tab1, tab2 = tabs

with tab_home:
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <h3 style="color: #0e1117; margin: 0;">🏠 홈 / 개요 <span style="background-color: #dee2e6; padding: 2px 8px; border-radius: 10px; font-size: 0.6em;
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ℹ️ 프로젝트 소개 및 데이터셋 설명", expanded=True):
        st.markdown("""
        **CMAPSS(Commercial Modular Aero-Propulsion System Simulation)** 데이터셋은 NASA에서 제공하는 항공기 엔진 고장 시뮬레이션 데이터입니다.
        - **목표**: 엔진의 센서 데이터를 분석하여 고장 전까지 남은 시간인 **RUL**(Remaining Useful Life)을 예측합니다.
        - **데이터 구성**:
            - **FD001 / FD003**: 단일 운전 조건 (Sea Level)
            - **FD002 / FD004**: 6가지 복합 운전 조건 (고도, 마하수 등 가변)
        """)
    
    st.subheader("📊 4개 데이터셋(FD001~004) 기본 통계 요약")
    stats_data = {
        "Dataset": ["FD001", "FD002", "FD003", "FD004"],
        "Train Units": [100, 260, 100, 249],
        "Test Units": [100, 259, 100, 248],
        "Conditions": ["1 (Sea Level)", "6 (Complex)", "1 (Sea Level)", "6 (Complex)"],
        "Fault Modes": ["HPC Degradation", "HPC Degradation", "HPC & Fan Degradation", "HPC & Fan Degradation"]
    }
    st.table(pd.DataFrame(stats_data))

    st.subheader("🏆 모델 성능 비교표 (최종 선정 결과)")
    comparison_data = {
        "Dataset" : ["FD001", "FD002", "FD003", "FD004"],
        "Model": ["LightGBM", "XGBoost", "XGBoost", "XGBoost"],
        "Best RMSE": ["14.01", "23.51", "13.10", "24.89"],
        "Best MAE" : ["10.69", "15.99", "10.01", "17.52"],
        "Best 나사_SCORE": ["185.5", "4043.55", "296.94", "3678.48"]
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

# [TAB 1] 모델 성능 분석
with tab1:
    st.header(f"🚀 {model_choice} 모델 성능 및 학습 지표")
    analysis_data = load_analysis_data(subset_choice, model_choice)
    
    if analysis_data and 'summary' in analysis_data:
        summary = analysis_data['summary']
        # 전체 지표 계산
        rmse = np.sqrt(((summary['actual_rul'] - summary['pred_rul']) ** 2).mean())
        mae = (summary['actual_rul'] - summary['pred_rul']).abs().mean()
        nasa_score = calculate_nasa_score(summary['actual_rul'], summary['pred_rul'])
        
        # 1. 모델 전체 성능 지표
        m1, m2, m3 = st.columns(3)
        m1.metric("전체 RMSE", f"{rmse:.2f}")
        m2.metric("전체 MAE", f"{mae:.2f}")
        m3.metric("NASA Score", f"{nasa_score:,.2f}")

        # 2. [위치 이동] 선택된 엔진(Unit)의 상세 요약 지표
        st.markdown("---")
        st.subheader(f"📍 Unit #{unit_id_input} 예측 요약")
        df_raw_mini = load_sensor_data(f"test_{subset_choice.lower()}", unit_id_input)
        if not df_raw_mini.empty:
            pred_rul, actual_rul = 0, 0
            target = summary[summary['unit_nr'] == unit_id_input]
            if not target.empty:
                pred_rul, actual_rul = target['pred_rul'].iloc[-1], target['actual_rul'].iloc[-1]
            
            u1, u2, u3 = st.columns(3)
            u1.metric("누적 사이클", f"{df_raw_mini['time_cycles'].max()}")
            u2.metric("실제 RUL", f"{int(actual_rul)}")
            u3.metric("예측 RUL", f"{int(pred_rul)}", delta=f"{int(pred_rul-actual_rul)}", delta_color="inverse")

        st.markdown("---")
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("📈 학습 곡선 (Loss Curve)")
            if 'history' in analysis_data:
                fig_loss = px.line(analysis_data['history'], x='epoch', y=['train_loss', 'val_loss'], template="plotly_dark")
                st.plotly_chart(fig_loss, use_container_width=True)
        with col_r:
            st.subheader("📊 엔진 전수 비교 (Actual vs Pred)")
            comp_df = summary.sort_values('unit_nr').drop_duplicates('unit_nr', keep='last')
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=comp_df['unit_nr'], y=comp_df['actual_rul'], mode='lines', name='Actual', line=dict(color='gray', dash='dash')))
            fig_comp.add_trace(go.Scatter(x=comp_df['unit_nr'], y=comp_df['pred_rul'], mode='lines+markers', name='Pred', line=dict(color='#FF6600')))
            fig_comp.update_layout(template="plotly_dark")
            st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("---")
        col_bl, col_br = st.columns([1.2, 0.8])
        with col_bl:
            st.subheader("🎯 실제 vs 예측 분포 (Scatter)")
            fig_scatter = px.scatter(summary, x='actual_rul', y='pred_rul', opacity=0.5, template="plotly_dark")
            max_v = max(summary['actual_rul'].max(), summary['pred_rul'].max())
            fig_scatter.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col_br:
            st.subheader("📄 상세 데이터 (엔진 50개)")
            st.dataframe(summary.head(50), use_container_width=True, height=400)

# [TAB 2] 센서 추세 분석
with tab2:
    st.header(f"🔎 Unit #{unit_id_input} 센서 분석")
    
    # [추가] 센서 명칭 가이드 (Expander로 깔끔하게 정리)
    with st.expander("📝 센서 번호별 명칭 안내", expanded=False):
        st.markdown("""
        | 센서 | Symbol | 명칭 (Description) | 단위 | | 센서 |  Symbol | 명칭 (Description) | 단위 |
        | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
        | **s_1** | T2 |Fan inlet temp(팬 입구 온도) | °R | | **s_12** | phi |Ratio of fuel flow to Ps30(연료/Ps30 비율) | pps/psia |
        | **s_2** | T24 |LPC outlet temp(LPC 출구 온도) | °R | | **s_13** | NRf |Corrected fan speed(보정 팬 속도) | rpm |
        | **s_3** | T30 |HPC outlet temp(HPC 출구 온도) | °R | | **s_14** | NRc |Corrected core speed(보정 코어속도) | rpm |
        | **s_4** | T50 |LPT outlet temp(LPT 출구 온도) | °R | | **s_15** | BPR |Bypass Ratio(바이패스 비율) | - |
        | **s_5** | P2 |Fan inlet Pressure(팬 입구 압력) | psia | | **s_16** | farB |Burner fuel-air ratio(버너 연료공기비)  | - |
        | **s_6** | P15 |bypass Pressure(바이패스 압력) | psia | | **s_17** | htBleed |Bleed Enthalpy(블리드 엔탈피) | - |
        | **s_7** | P30 |Total HPC outlet pressure(HPC 출구 압력) | psia | | **s_18** | Nf_dmd |Demanded fan speed(요구 팬 속도) | rpm |
        | **s_8** | Nf |Physical fan speed(팬 속도) | rpm | | **s_19** | PCNfR_dmd |Demanded core speed(요구 코어 속도) | rpm |
        | **s_9** | Nc |Physical core speed(코어 속도) | rpm | | **s_20** | W31 |HPT coolant bleed(HPT 냉각블리드) | lbm/s |
        | **s_10** | epr |Engine pressure ratio(엔진 압력비) | - | | **s_21** | W32 |LPT coolant bleed(LPT 냉각블리드) | lbm/s |
        | **s_11** | Ps30 |HPC outlet static pressure(HPC 출구 정적압력) | psia | | - | - | - | - |           
        """)

    useful_map = {
        "FD001": [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21], 
        "FD002": [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21], 
        "FD003": [2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21], 
        "FD004": [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 21]
    }
    current_useful = [f"s_{i}" for i in useful_map.get(subset_choice, [2, 3, 4])]
    cluster_map = {
        0: "최대 고도 고속 (42k, 0.84M)", 1: "지상 및 이륙 (0ft, 0M)", 2: "중고도 비행 (20k, 0.7M)",
        3: "중고도 가변 (25k, 0.6M)", 4: "저고도 접근 (10k, 0.25M)", 5: "일반 순항 (35k, 0.84M)"
    }
    
    try:
        df_raw = load_sensor_data(f"test_{subset_choice.lower()}", unit_id_input)
        df_prep = load_sensor_data(f"test_{subset_choice.lower()}_prep", unit_id_input)
        
        if not df_raw.empty:
            # 탭 2 상단 메트릭은 제거됨 (탭 1로 이동)
            st.markdown("---")
            sel_col, guide_col = st.columns([1.5, 1])
            all_s = sorted([c for c in df_raw.columns if re.match(r'^s_\d+$', c)], key=lambda x: int(x.split('_')[1]))
            with sel_col: selected = st.multiselect("센서 선택", all_s, default=current_useful[:3])
            with guide_col:
                st.markdown(f"""<div style="background-color: #1e2130; padding: 12px; border-radius: 8px; border-left: 5px solid #FF4B4B;">
                    <span style="font-weight: bold; color: #FF4B4B;">💡 {subset_choice} 가이드</span><br>
                    추천 지표: <code>{', '.join(current_useful[:])}</div>""", unsafe_allow_html=True)
            
            st.markdown("---")
            c_left, c_right = st.columns(2)
            with c_left:
                st.subheader("📡 Raw Sensor Trend")
                st.plotly_chart(px.line(df_raw, x='time_cycles', y=selected, template="plotly_dark"), use_container_width=True)
            with c_right:
                st.subheader("🧪 Processed Trend (with Cluster)")
                if not df_prep.empty:
                    avail_prep = [s for s in selected if s in df_prep.columns]
                    color_col = None
                    if subset_choice in ["FD002", "FD004"]:
                        if 'op_cluster' in df_prep.columns:
                            df_prep['Cluster_Name'] = df_prep['op_cluster'].map(cluster_map).fillna("기타")
                            color_col = 'Cluster_Name'
                    else:
                        df_prep['Condition'] = "단일 운행 조건 (Single)"
                        color_col = 'Condition'
                    fig_prep = px.line(df_prep, x='time_cycles', y=avail_prep, color=color_col, template="plotly_dark")
                    st.plotly_chart(fig_prep, use_container_width=True)
                else: st.warning("전처리 데이터 없음")
            
            st.markdown("---")
            st.subheader(f"🌡️ 상관관계 (Unit #{unit_id_input})")
            if not df_prep.empty:
                df_corr = df_prep.merge(df_raw[['time_cycles', 'true_rul']], on='time_cycles', how='left')
                v_cols = [s for s in selected if s in df_corr.columns] + ['true_rul']
                st.plotly_chart(px.imshow(df_corr[v_cols].corr(), text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark"), use_container_width=True)
                st.caption("💡 **분석 팁**: 값이 1에 가까울수록 RUL과 정비례(같이 증가), -1에 가까울수록 반비례(수명 감소 시 센서값 증가) 관계임을 나타냅니다.")
    except Exception as e: st.error(f"오류: {e}")
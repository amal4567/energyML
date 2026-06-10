import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="EnergyML Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CLEAN PROFESSIONAL DESIGN — LIGHT THEME
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ─── VARIABLES ─── */
:root {
    --bg:           #F5F6FA;
    --surface:      #FFFFFF;
    --surface-2:    #F0F2F8;
    --border:       #E2E5F0;
    --border-soft:  #ECEEF5;

    --indigo:       #4361EE;
    --indigo-light: #EEF1FD;
    --indigo-mid:   #7B93F5;
    --indigo-dark:  #2B47C8;

    --cyan:         #0EA5E9;
    --emerald:      #10B981;
    --amber:        #F59E0B;
    --rose:         #F43F5E;

    --ink:          #1A1D2E;
    --ink-2:        #3D4166;
    --muted:        #8B90A7;
    --muted-light:  #B8BDD4;

    --mono:         'DM Mono', monospace;
    --display:      'Syne', sans-serif;
    --body:         'DM Sans', sans-serif;

    --radius:       10px;
    --radius-lg:    16px;
    --shadow-sm:    0 1px 3px rgba(67,97,238,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow:       0 4px 16px rgba(67,97,238,0.08), 0 1px 4px rgba(0,0,0,0.04);
    --shadow-lg:    0 12px 40px rgba(67,97,238,0.12), 0 4px 12px rgba(0,0,0,0.06);
}

/* ─── BASE ─── */
* { box-sizing: border-box; }

.stApp {
    background: var(--bg);
    color: var(--ink);
    font-family: var(--body);
}

/* ─── SIDEBAR ─── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 24px rgba(67,97,238,0.06);
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500 !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-family: var(--display) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: var(--ink) !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 2px solid var(--indigo-light) !important;
    padding-bottom: 0.6rem !important;
    margin-bottom: 1rem !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] select {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    border-radius: var(--radius) !important;
}

section[data-testid="stSidebar"] input:focus,
section[data-testid="stSidebar"] select:focus {
    border-color: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(67,97,238,0.12) !important;
    outline: none !important;
}

/* Slider */
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--indigo) !important;
    box-shadow: 0 0 0 3px rgba(67,97,238,0.2) !important;
}

/* ─── HEADER ─── */
.eng-header {
    display: flex;
    align-items: center;
    gap: 1.4rem;
    padding: 2rem 0 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}

.eng-logo {
    width: 52px;
    height: 52px;
    background: linear-gradient(135deg, var(--indigo), var(--indigo-dark));
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
    box-shadow: 0 6px 20px rgba(67,97,238,0.35);
}

.eng-title-block { display: flex; flex-direction: column; gap: 0.25rem; }

.eng-main-title {
    font-family: var(--display);
    font-size: 2rem;
    font-weight: 800;
    color: var(--ink);
    letter-spacing: -0.01em;
    line-height: 1;
}

.eng-main-title span { color: var(--indigo); }

.eng-system-id {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.eng-status-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.15rem;
}

.eng-status-dot {
    width: 7px; height: 7px;
    background: var(--emerald);
    border-radius: 50%;
    box-shadow: 0 0 6px var(--emerald);
    animation: pulse-ok 2.5s ease-in-out infinite;
}

@keyframes pulse-ok {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.eng-status-text {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--emerald);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid var(--border) !important;
    gap: 0.25rem;
}

.stTabs [data-baseweb="tab"] {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    padding: 0.65rem 1.4rem !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px;
    transition: all 0.2s;
    border-radius: 6px 6px 0 0 !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--indigo) !important;
    background: var(--indigo-light) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--indigo) !important;
    border-bottom-color: var(--indigo) !important;
    background: var(--indigo-light) !important;
    font-weight: 500 !important;
}

.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ─── SECTION LABELS ─── */
.section-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted-light);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}

.section-label::before {
    content: '';
    width: 3px; height: 12px;
    background: var(--indigo);
    border-radius: 2px;
    flex-shrink: 0;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border-soft);
}

/* ─── METRIC CARDS ─── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.metric-cell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}

.metric-cell::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--border-soft);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}

.metric-cell:hover {
    box-shadow: var(--shadow);
    transform: translateY(-1px);
}

.metric-cell.primary::after {
    background: linear-gradient(90deg, var(--indigo), var(--indigo-mid));
}

.metric-cell .m-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.6rem;
}

.metric-cell .m-value {
    font-family: var(--display);
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--ink);
    line-height: 1;
    letter-spacing: -0.02em;
}

.metric-cell.primary .m-value {
    color: var(--indigo);
}

.metric-cell .m-unit {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted-light);
    margin-top: 0.35rem;
}

/* ─── PREDICT BUTTON ─── */
.stButton > button {
    background: var(--indigo) !important;
    border: none !important;
    color: #FFFFFF !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 2.2rem !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 4px 14px rgba(67,97,238,0.35) !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: var(--indigo-dark) !important;
    box-shadow: 0 6px 20px rgba(67,97,238,0.45) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0px) scale(0.98) !important;
}

/* ─── ALERTS ─── */
.stAlert {
    background: rgba(16,185,129,0.06) !important;
    border: 1px solid rgba(16,185,129,0.25) !important;
    border-radius: var(--radius) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    color: #059669 !important;
    letter-spacing: 0.04em;
}

/* ─── DATAFRAME ─── */
.dataframe, [data-testid="stDataFrame"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    box-shadow: var(--shadow-sm);
}

/* ─── HISTORY PANEL ─── */
.history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.9rem 1.2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: var(--radius) var(--radius) 0 0;
    margin-top: 1.5rem;
}

.history-title {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--indigo);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
}

.history-count {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted-light);
    letter-spacing: 0.08em;
    background: var(--indigo-light);
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    color: var(--indigo);
}

/* ─── HEADINGS ─── */
h1, h2, h3 {
    font-family: var(--display) !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ink) !important;
}

h3 { font-size: 1rem !important; color: var(--ink-2) !important; }

/* ─── CHART CONTAINERS ─── */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
    background: var(--surface);
    box-shadow: var(--shadow-sm);
}

/* ─── HIDE STREAMLIT CHROME ─── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--indigo-mid); }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA & MODEL
# =====================================================
model = joblib.load("models/best_model.pkl")

model = joblib.load("models/best_model.pkl")

X = df.drop(columns=["meter_reading", "log_meter_reading"])

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="eng-header">
    <div class="eng-logo">⚡</div>
    <div class="eng-title-block">
        <div class="eng-main-title">Energy<span>ML</span> Dashboard</div>
        <div class="eng-system-id">SYS-ID: EML-v2.1 · ASHRAE Energy Prediction System</div>
        <div class="eng-status-row">
            <div class="eng-status-dot"></div>
            <div class="eng-status-text">All systems nominal · Model loaded</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR INPUTS
# =====================================================
st.sidebar.header("Building Information")

building_id = st.sidebar.number_input("Building ID", value=100)
meter = st.sidebar.selectbox("Meter Type", [0, 1, 2, 3])
site_id = st.sidebar.number_input("Site ID", value=1)

primary_use = st.sidebar.selectbox("Building Type", [
    "Entertainment/public assembly",
    "Food sales and service",
    "Healthcare",
    "Lodging/residential",
    "Manufacturing/industrial",
    "Office",
    "Other",
    "Parking",
    "Public services",
    "Religious worship",
    "Retail",
    "Services",
    "Technology/science",
    "Utility",
    "Warehouse/storage"
])

square_feet = st.sidebar.number_input("Square Feet", 100, 200000, 5000)
year_built = st.sidebar.number_input("Year Built", 1900, 2025, 2000)

air_temperature = st.sidebar.slider("Air Temperature", -30.0, 50.0, 20.0)
cloud_coverage = st.sidebar.slider("Cloud Coverage", 0.0, 10.0, 5.0)
dew_temperature = st.sidebar.slider("Dew Temperature", -30.0, 40.0, 10.0)
precip_depth_1_hr = st.sidebar.slider("Precipitation", 0.0, 100.0, 0.0)
sea_level_pressure = st.sidebar.slider("Sea Pressure", 900.0, 1100.0, 1013.0)
wind_direction = st.sidebar.slider("Wind Direction", 0, 360, 180)
wind_speed = st.sidebar.slider("Wind Speed", 0.0, 30.0, 5.0)

month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.slider("Day", 1, 31, 15)
hour = st.sidebar.slider("Hour", 0, 23, 12)
weekday = st.sidebar.slider("Weekday", 0, 6, 2)

# =====================================================
# FEATURE ENGINEERING
# =====================================================
is_weekend = 1 if weekday >= 5 else 0
building_age = 2025 - year_built
log_square_feet = np.log1p(square_feet)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(
    ["⚡  Prediction", "◈  Feature Importance", "▦  Dataset"])

# =====================================================
# TAB 1 - PREDICTION
# =====================================================
with tab1:

    st.markdown('<div class="section-label">Inference Panel</div>',
                unsafe_allow_html=True)

    input_data = {col: 0 for col in X.columns}
    input_data.update({
        "building_id": building_id,
        "meter": meter,
        "site_id": site_id,
        "square_feet": square_feet,
        "year_built": year_built,
        "air_temperature": air_temperature,
        "cloud_coverage": cloud_coverage,
        "dew_temperature": dew_temperature,
        "precip_depth_1_hr": precip_depth_1_hr,
        "sea_level_pressure": sea_level_pressure,
        "wind_direction": wind_direction,
        "wind_speed": wind_speed,
        "month": month,
        "day": day,
        "hour": hour,
        "weekday": weekday,
        "is_weekend": is_weekend,
        "building_age": building_age,
        "log_square_feet": log_square_feet
    })

    selected_col = f"primary_use_{primary_use}"
    if selected_col in input_data:
        input_data[selected_col] = 1

    input_df = pd.DataFrame([input_data])

    if st.button("▶  RUN PREDICTION"):

        prediction_log = model.predict(input_df)[0]
        prediction = np.expm1(prediction_log)

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-cell primary">
                <div class="m-label">Energy Consumption</div>
                <div class="m-value">{prediction:.1f}</div>
                <div class="m-unit">kWh · predicted output</div>
            </div>
            <div class="metric-cell">
                <div class="m-label">Floor Area</div>
                <div class="m-value">{square_feet:,}</div>
                <div class="m-unit">ft² · gross area</div>
            </div>
            <div class="metric-cell">
                <div class="m-label">Building Age</div>
                <div class="m-value">{building_age}</div>
                <div class="m-unit">years · since construction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # GAUGE — clean light theme
        gauge_max = max(prediction * 2, 100)
        pct = prediction / gauge_max

        if pct < 0.4:
            bar_color = "#10B981"
        elif pct < 0.75:
            bar_color = "#F59E0B"
        else:
            bar_color = "#F43F5E"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={
                "font": {"family": "DM Mono", "color": "#4361EE", "size": 36},
                "suffix": " kWh"
            },
            title={
                "text": "ENERGY OUTPUT",
                "font": {"family": "Syne", "color": "#8B90A7", "size": 12}
            },
            gauge={
                "axis": {
                    "range": [0, gauge_max],
                    "tickfont": {"family": "DM Mono", "color": "#B8BDD4", "size": 10},
                    "tickcolor": "#E2E5F0",
                },
                "bar": {"color": bar_color, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, gauge_max * 0.4],
                        "color": "rgba(16,185,129,0.07)"},
                    {"range": [gauge_max * 0.4, gauge_max * 0.75],
                        "color": "rgba(245,158,11,0.07)"},
                    {"range": [gauge_max * 0.75, gauge_max],
                        "color": "rgba(244,63,94,0.07)"},
                ],
                "threshold": {
                    "line": {"color": bar_color, "width": 2},
                    "thickness": 0.85,
                    "value": prediction
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            font_color="#8B90A7",
            margin=dict(t=60, b=20, l=30, r=30),
            height=280,
        )

        st.plotly_chart(fig, use_container_width=True)

        # History
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Type": primary_use,
            "Area (ft²)": square_feet,
            "Prediction (kWh)": round(prediction, 2)
        })

        st.success("✓  Inference complete — results stored in session log")

    # History table
    if "history" in st.session_state and len(st.session_state.history) > 0:
        hist_df = pd.DataFrame(st.session_state.history)
        count = len(hist_df)

        st.markdown(f"""
        <div class="history-header">
            <div class="history-title">◈ Session Log</div>
            <div class="history-count">{count} record{'s' if count != 1 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(hist_df, use_container_width=True, hide_index=True)

# =====================================================
# TAB 2 - FEATURE IMPORTANCE
# =====================================================
with tab2:

    st.markdown('<div class="section-label">Model Explainability · Feature Weights</div>',
                unsafe_allow_html=True)

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.dataframe(importance.head(
            15), use_container_width=True, hide_index=True)

        top10 = importance.head(10).iloc[::-1]

        mpl.rcParams.update({
            'figure.facecolor': '#FFFFFF',
            'axes.facecolor':   '#FFFFFF',
            'axes.edgecolor':   '#E2E5F0',
            'axes.labelcolor':  '#8B90A7',
            'xtick.color':      '#B8BDD4',
            'ytick.color':      '#3D4166',
            'text.color':       '#3D4166',
            'grid.color':       '#F0F2F8',
            'font.family':      'monospace',
            'font.size':        9,
        })

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor('#FFFFFF')

        colors = ['#4361EE' if i == len(top10) - 1 else '#7B93F5'
                  for i in range(len(top10))]
        alphas = [1.0 if i == len(top10) - 1 else 0.65
                  for i in range(len(top10))]

        bars = ax.barh(
            top10["Feature"],
            top10["Importance"],
            color=colors,
            alpha=1.0,
            height=0.55,
            edgecolor='none'
        )
        for bar, alpha in zip(bars, alphas):
            bar.set_alpha(alpha)

        ax.set_xlabel("Importance Score", labelpad=8, fontsize=8,
                      color='#B8BDD4', fontfamily='monospace')
        ax.set_title("TOP 10 PREDICTORS", fontsize=9, color='#8B90A7',
                     fontfamily='monospace', fontweight='normal', loc='left', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E2E5F0')
        ax.spines['bottom'].set_color('#E2E5F0')
        ax.xaxis.grid(True, alpha=0.6, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{w:.4f}', va='center', ha='left',
                    fontsize=7.5, color='#B8BDD4', fontfamily='monospace')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# =====================================================
# TAB 3 - DATASET
# =====================================================
with tab3:

    st.markdown('<div class="section-label">Dataset Overview · Raw Records</div>',
                unsafe_allow_html=True)

    r, c = df.shape
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(f"""
        <div class="metric-cell" style="border-radius:12px;border:1px solid #E2E5F0;padding:1.2rem;background:#fff;box-shadow:0 1px 4px rgba(67,97,238,0.06)">
            <div class="m-label">Total Rows</div>
            <div class="m-value" style="font-size:1.8rem;color:#4361EE">{r:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="metric-cell" style="border-radius:12px;border:1px solid #E2E5F0;padding:1.2rem;background:#fff;box-shadow:0 1px 4px rgba(67,97,238,0.06)">
            <div class="m-label">Features</div>
            <div class="m-value" style="font-size:1.8rem;color:#4361EE">{c}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div class="metric-cell" style="border-radius:12px;border:1px solid #E2E5F0;padding:1.2rem;background:#fff;box-shadow:0 1px 4px rgba(67,97,238,0.06)">
            <div class="m-label">Memory</div>
            <div class="m-value" style="font-size:1.8rem;color:#4361EE">{df.memory_usage(deep=True).sum() / 1e6:.1f}</div>
            <div class="m-unit">MB</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Sample — first 20 rows</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">Descriptive Statistics</div>',
                unsafe_allow_html=True)
    st.dataframe(df.describe(), use_container_width=True)

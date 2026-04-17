# ================================================================
#  APL LOGISTICS — DELIVERY RISK INTELLIGENCE DASHBOARD
# ================================================================
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import base64
import os

import encoders
import data_processing

# FORCE pickle compatibility (VERY IMPORTANT)
sys.modules["encoders"] = encoders
sys.modules["data_processing"] = data_processing

warnings.filterwarnings("ignore")

# ================================================================
# 0 — PAGE CONFIG & THEME
# ================================================================

st.set_page_config(
    page_title="APL Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)




# ---------------------------------------------------
# HEADER LOGOS
# ---------------------------------------------------


def get_base64_image(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

logo_base64 = get_base64_image("assets/logo.png")

st.markdown(
    f"""
    <div style=" background-color: #0D1117; 
    border-bottom: 1px solid #e5e7eb; 
    padding: 25px 0; 
    display: flex; 
    justify-content: left; 
    align-items: center; gap: 8vw; ">
        <img src="data:image/png;base64,{logo_base64}" 
        style="max-height: 120px; width: auto;" >
    </div>
    """,
    unsafe_allow_html=True
)



# ── Dark theme injection ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background ───────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #0D1117 !important;
    color: #E6EDF3 !important;
}

[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #21262D;
}

[data-testid="stSidebarContent"] * {
    color: #C9D1D9 !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #161B22;
    border-bottom: 1px solid #21262D;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 6px 6px 0 0;
    color: #8B949E;
    font-size: 13px;
    font-weight: 500;
    padding: 10px 20px;
    letter-spacing: 0.4px;
}
.stTabs [aria-selected="true"] {
    background: #0D1117;
    color: #58A6FF !important;
    border-bottom: 2px solid #58A6FF;
}

/* ── KPI Cards ───────────────────────────────────────────── */
.kpi-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}
.kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #8B949E;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 12px;
    color: #8B949E;
    margin-top: 4px;
}
.kpi-red   { color: #FF6B6B; }
.kpi-amber { color: #E3A945; }
.kpi-green { color: #3FB950; }
.kpi-blue  { color: #58A6FF; }

/* ── Section headers ─────────────────────────────────────── */
.section-header {
    font-size: 13px;
    font-weight: 600;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #21262D;
    padding-bottom: 8px;
    margin: 24px 0 16px;
}

/* ── Risk badges ─────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-high   { background: rgba(255,107,107,0.15); color: #FF6B6B; border: 1px solid rgba(255,107,107,0.3); }
.badge-medium { background: rgba(227,169,69,0.15);  color: #E3A945; border: 1px solid rgba(227,169,69,0.3);  }
.badge-low    { background: rgba(63,185,80,0.15);   color: #3FB950; border: 1px solid rgba(63,185,80,0.3);   }

/* ── Dataframe ───────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #21262D;
    border-radius: 8px;
}

/* ── Inputs ──────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div,
[data-testid="stSlider"],
[data-testid="stNumberInput"] {
    background: #161B22 !important;
}

/* ── Dividers ────────────────────────────────────────────── */
hr { border-color: #21262D !important; }

/* ── Score gauge container ───────────────────────────────── */
.score-box {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 12px;
    padding: 28px;
    text-align: center;
}
.score-num {
    font-size: 52px;
    font-weight: 800;
    letter-spacing: -1px;
}
.score-label {
    font-size: 13px;
    color: #8B949E;
    margin-top: 4px;
}
/* ── Custom Primary Button ───────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #FF4B4B, #E03E3E);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.4px;
    transition: all 0.2s ease-in-out;
}

/* Hover effect */
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(88,166,255,0.35);
}

/* Click effect */
.stButton > button:active {
    transform: scale(0.98);
}
</style>
""", unsafe_allow_html=True)

# Plotly dark template applied to all charts
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG       = "#161B22"
PLOTLY_PAPER_BG = "#0D1117"
PLOTLY_GRID     = "#21262D"
COLOR_HIGH      = "#FF6B6B"
COLOR_MED       = "#E3A945"
COLOR_LOW       = "#3FB950"
COLOR_BLUE      = "#58A6FF"

RISK_COLORS = {
    "High":   COLOR_HIGH,
    "Medium": COLOR_MED,
    "Low":    COLOR_LOW,
}

def apply_dark_layout(fig, height=420, legend=True):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_BG,
        font=dict(family="monospace", color="#C9D1D9", size=12),
        height=height,
        margin=dict(l=16, r=16, t=36, b=16),
        showlegend=legend,
        legend=dict(
            bgcolor="#161B22",
            bordercolor="#21262D",
            borderwidth=1,
            font=dict(size=11),
        ) if legend else {},
        xaxis=dict(gridcolor=PLOTLY_GRID, linecolor="#21262D", zerolinecolor=PLOTLY_GRID),
        yaxis=dict(gridcolor=PLOTLY_GRID, linecolor="#21262D", zerolinecolor=PLOTLY_GRID),
    )
    return fig


# ================================================================
# 1 — DATA & MODEL LOADING
# ================================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        df = pd.read_csv("data/all_orders_scored.csv")
        df["Risk_Category"] = pd.Categorical(
            df["Risk_Category"], categories=["Low", "Medium", "High"], ordered=True
        )
        # Ensure Data_Split column exists even if loading an older file
        if "Data_Split" not in df.columns:
            df["Data_Split"] = "All"
        return df
    except FileNotFoundError:
        # Graceful fallback to test-only file if full scored file not yet generated
        try:
            df = pd.read_csv("data/scored_test_data.csv")
            df["Risk_Category"] = pd.Categorical(
                df["Risk_Category"], categories=["Low", "Medium", "High"], ordered=True
            )
            df["Data_Split"] = "Test"
            st.warning(
                "⚠️  `data/all_orders_scored.csv` not found — showing test set only "
                f"({len(df):,} orders). Re-run `apl_pipeline_v2.py` to score all orders."
            )
            return df
        except FileNotFoundError:
            st.error(
                "❌  No scored data found. Run `apl_pipeline_v2.py` first to "
                "generate `data/all_orders_scored.csv`."
            )
            st.stop()


@st.cache_resource
def load_artifacts():
    try:
        base_dir = os.path.dirname(__file__)

        model = joblib.load(os.path.join(base_dir, "models/risk_model.pkl"))
        encoder = joblib.load(os.path.join(base_dir, "models/target_encoder.pkl"))
        stats = joblib.load(os.path.join(base_dir, "models/train_stats.pkl"))
        quantiles = joblib.load(os.path.join(base_dir, "models/value_quantiles.pkl"))

        ohe_cols = pd.read_csv(
            os.path.join(base_dir, "data/ohe_columns.csv"),
            header=None
        )[0].tolist()

        return model, encoder, stats, quantiles, ohe_cols

    except FileNotFoundError:
        return None, None, None, None, None


df_full  = load_data()
artifacts = load_artifacts()
MODEL_LOADED = artifacts[0] is not None

# ================================================================
# 2 — SIDEBAR FILTERS
# ================================================================

with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px'>
      <div style='font-size:11px;color:#8B949E;letter-spacing:1.5px;
                  text-transform:uppercase;margin-bottom:4px'>
        APL Logistics
      </div>
      <div style='font-size:18px;font-weight:700;color:#E6EDF3'>
        Risk Intelligence
      </div>
      <div style='font-size:11px;color:#58A6FF;margin-top:2px'>
        OPERATIONS DASHBOARD
      </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='section-header'>Global Filters</div>",
        unsafe_allow_html=True,
    )

    # Risk threshold slider
    risk_threshold = st.slider(
        "Risk threshold",
        min_value=0.10, max_value=0.90, value=0.35, step=0.05,
        help="Orders above this probability are flagged as high-risk",
    )

    # Shipping mode
    all_modes = sorted(df_full["Shipping Mode"].dropna().unique().tolist())
    sel_modes = st.multiselect(
        "Shipping mode",
        options=all_modes,
        default=all_modes,
    )

    # Market / Region
    region_col = next(
        (c for c in ["Order Region", "Market"] if c in df_full.columns), None
    )
    if region_col:
        all_regions = sorted(df_full[region_col].dropna().unique().tolist())
        sel_regions = st.multiselect(
            "Region / Market",
            options=all_regions,
            default=all_regions,
        )
    else:
        sel_regions = []

    # Customer segment
    if "Customer Segment" in df_full.columns:
        all_segs = sorted(df_full["Customer Segment"].dropna().unique().tolist())
        sel_segs = st.multiselect(
            "Customer segment",
            options=all_segs,
            default=all_segs,
        )
    else:
        sel_segs = []

    # Data split selector — lets analysts isolate test set for audit
    all_splits = sorted(df_full["Data_Split"].dropna().unique().tolist())
    if len(all_splits) > 1:
        st.markdown(
            "<div class='section-header' style='margin-top:16px'>Data View</div>",
            unsafe_allow_html=True,
        )
        sel_splits = st.multiselect(
            "Data split",
            options=all_splits,
            default=all_splits,
            help=(
                "Train+Test = full 180k fleet view (default for KPIs). "
                "Select Test only to audit model on held-out data."
            ),
        )
    else:
        sel_splits = all_splits

    st.markdown("<hr>", unsafe_allow_html=True)
    total_full = len(df_full)
    st.markdown(
        f"<div style='font-size:11px;color:#8B949E'>"
        f"Total orders in dataset: <b style='color:#C9D1D9'>{total_full:,}</b><br>"
        f"Model loaded: {'✅' if MODEL_LOADED else '⚠️ Not found'}</div>",
        unsafe_allow_html=True,
    )


# Apply filters
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    if sel_modes and "Shipping Mode" in df.columns:
        mask &= df["Shipping Mode"].isin(sel_modes)
    if sel_regions and region_col and region_col in df.columns:
        mask &= df[region_col].isin(sel_regions)
    if sel_segs and "Customer Segment" in df.columns:
        mask &= df["Customer Segment"].isin(sel_segs)
    if sel_splits and "Data_Split" in df.columns:
        mask &= df["Data_Split"].isin(sel_splits)
    # Re-apply threshold to Risk_Category
    df = df[mask].copy()
    df["Risk_Category"] = pd.cut(
        df["Late_Delivery_Probability"],
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    )
    return df


df = apply_filters(df_full)

# ================================================================
# 3 — HEADER BAR
# ================================================================

st.markdown("""
<div style='display:flex;align-items:baseline;gap:12px;margin-bottom:8px'>
  <span style='font-size:26px;font-weight:800;letter-spacing:-0.5px'>
    Machine Learning–Based Late Delivery Risk Prediction in Global Supply Chain Operations
  </span>
  <span style='font-size:13px;color:#58A6FF;font-family:monospace'>
    LIVE ANALYTICS
  </span>
</div>
""", unsafe_allow_html=True)

# ── Top KPI bar ──────────────────────────────────────────────────
total_orders   = len(df)
total_unfiltered = len(df_full)
high_risk      = (df["Risk_Category"] == "High").sum()
med_risk       = (df["Risk_Category"] == "Medium").sum()
low_risk       = (df["Risk_Category"] == "Low").sum()
avg_prob       = df["Late_Delivery_Probability"].mean() * 100

# Label clarifies whether filters are narrowing the view
is_filtered = total_orders < total_unfiltered
scope_label = (
    f"of {total_unfiltered:,} total"
    if is_filtered
    else "full fleet"
)

if "Actual_Late" in df.columns:
    actual_rate = df["Actual_Late"].mean() * 100
    recall_at_thr = (
        df[df["Actual_Late"] == 1]["Late_Delivery_Probability"] >= risk_threshold
    ).mean() * 100
else:
    actual_rate = None

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>Total Orders</div>
      <div class='kpi-value kpi-blue'>{total_orders:,}</div>
      <div class='kpi-sub'>{scope_label}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>High Risk Orders</div>
      <div class='kpi-value kpi-red'>{high_risk:,}</div>
      <div class='kpi-sub'>{high_risk/total_orders*100:.1f}% · need intervention</div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>Medium Risk Orders</div>
      <div class='kpi-value kpi-amber'>{med_risk:,}</div>
      <div class='kpi-sub'>{med_risk/total_orders*100:.1f}% · monitor closely</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>Avg Risk Probability</div>
      <div class='kpi-value kpi-amber'>{avg_prob:.1f}<span style='font-size:18px'>%</span></div>
      <div class='kpi-sub'>late delivery probability</div>
    </div>""", unsafe_allow_html=True)

with k5:
    if actual_rate is not None:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Actual Delay Rate</div>
          <div class='kpi-value kpi-red'>{actual_rate:.1f}<span style='font-size:18px'>%</span></div>
          <div class='kpi-sub'>ground truth</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>Low Risk Orders</div>
          <div class='kpi-value kpi-green'>{low_risk:,}</div>
          <div class='kpi-sub'>{low_risk/total_orders*100:.1f}% of orders</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ================================================================
# 4 — TABS
# ================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Delay Risk Overview",
    "Order-Level Risk Prediction",
    "Region & Mode Risk Analysis",
    "Operations Action Panel",
])

# ================================================================
# TAB 1 — DELAY RISK OVERVIEW
# ================================================================

with tab1:
    st.markdown(
        "<div class='section-header'>Overall Risk Distribution</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        # Donut — risk category share
        cat_counts = df["Risk_Category"].value_counts().reindex(["High", "Medium", "Low"])
        fig = go.Figure(go.Pie(
            labels=cat_counts.index.tolist(),
            values=cat_counts.values.tolist(),
            hole=0.62,
            marker=dict(
                colors=[COLOR_HIGH, COLOR_MED, COLOR_LOW],
                line=dict(color=PLOTLY_PAPER_BG, width=3),
            ),
            textinfo="label+percent",
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>Orders: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig.add_annotation(
            text=f"<b>{total_orders:,}</b><br><span style='font-size:11px'>Orders</span>",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=22, color="#E6EDF3"),
        )
        fig.update_layout(
            title="Risk Category Breakdown",
            paper_bgcolor=PLOTLY_PAPER_BG,
            plot_bgcolor=PLOTLY_BG,
            font=dict(color="#C9D1D9", size=12),
            height=380,
            margin=dict(l=16, r=16, t=44, b=16),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Probability distribution histogram
        fig = go.Figure()
        for cat, color in [("High", COLOR_HIGH), ("Medium", COLOR_MED), ("Low", COLOR_LOW)]:
            sub = df[df["Risk_Category"] == cat]["Late_Delivery_Probability"]
            fig.add_trace(go.Histogram(
                x=sub, name=cat,
                nbinsx=40,
                marker_color=color,
                opacity=0.75,
                histnorm="probability density",
                hovertemplate=f"<b>{cat}</b><br>Probability: %{{x:.2f}}<br>Density: %{{y:.3f}}<extra></extra>",
            ))
        fig.add_vline(
            x=risk_threshold, line_dash="dot",
            line_color="#FFFFFF", line_width=1.5,
            annotation_text=f"Threshold {risk_threshold}",
            annotation_font_color="#FFFFFF",
            annotation_font_size=11,
        )
        fig = apply_dark_layout(fig, height=380)
        fig.update_layout(
            title="Predicted Risk Probability Distribution",
            barmode="overlay",
            xaxis_title="Late Delivery Probability",
            yaxis_title="Density",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='section-header'>Risk by Key Dimensions</div>",
        unsafe_allow_html=True,
    )

    c3, c4 = st.columns([1, 1])

    with c3:
        # Box plot — probability by shipping mode
        if "Shipping Mode" in df.columns:
            fig = px.box(
                df, x="Shipping Mode", y="Late_Delivery_Probability",
                color="Shipping Mode",
                color_discrete_map={
                    m: c for m, c in zip(
                        sorted(df["Shipping Mode"].unique()),
                        [COLOR_BLUE, COLOR_LOW, COLOR_MED, COLOR_HIGH],
                    )
                },
                labels={"Late_Delivery_Probability": "Risk Probability"},
                template=PLOTLY_TEMPLATE,
            )
            fig = apply_dark_layout(fig, height=360, legend=False)
            fig.update_layout(title="Risk Probability by Shipping Mode",
                              xaxis_title="", yaxis_title="Probability")
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        # Bar — actual vs predicted delay rate by segment
        if "Customer Segment" in df.columns and "Actual_Late" in df.columns:
            seg_stats = (
                df.groupby("Customer Segment")
                  .agg(
                      Predicted=("Late_Delivery_Probability", "mean"),
                      Actual=("Actual_Late", "mean"),
                  )
                  .reset_index()
            )
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Predicted avg prob",
                x=seg_stats["Customer Segment"],
                y=seg_stats["Predicted"],
                marker_color=COLOR_BLUE, opacity=0.85,
                hovertemplate="<b>%{x}</b><br>Predicted: %{y:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                name="Actual delay rate",
                x=seg_stats["Customer Segment"],
                y=seg_stats["Actual"],
                marker_color=COLOR_HIGH, opacity=0.85,
                hovertemplate="<b>%{x}</b><br>Actual: %{y:.3f}<extra></extra>",
            ))
            fig = apply_dark_layout(fig, height=360)
            fig.update_layout(
                title="Predicted vs Actual Delay Rate by Segment",
                barmode="group",
                xaxis_title="",
                yaxis_title="Rate",
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, use_container_width=True)


# ================================================================
# TAB 2 — ORDER-LEVEL RISK PREDICTION
# ================================================================

with tab2:
    st.markdown(
        "<div class='section-header'>Score a New Order</div>",
        unsafe_allow_html=True,
    )

    if not MODEL_LOADED:
        st.warning(
            "⚠️  Model artifacts not found. "
            "Run `apl_pipeline_v2.py` to generate models/ and data/ folders."
        )
    else:
        model, encoder, train_stats, value_quantiles, ohe_cols = artifacts

        # Collect order inputs
        with st.container():
            f1, f2, f3 = st.columns(3)
            with f1:
                shipping_mode = st.selectbox(
                    "Shipping Mode",
                    options=["Standard Class", "Second Class", "First Class", "Same Day"],
                )
                order_qty = st.number_input(
                    "Order Item Quantity", min_value=1, max_value=200, value=5
                )
                scheduled_days = st.number_input(
                    "Scheduled Shipping Days", min_value=1, max_value=30, value=4
                )
            with f2:
                order_region = st.selectbox(
                    "Order Region",
                    options=sorted(df_full[region_col].dropna().unique()) if region_col else ["Unknown"],
                )
                customer_segment = st.selectbox(
                    "Customer Segment",
                    options=["Consumer", "Corporate", "Home Office"],
                )
                market = st.selectbox(
                    "Market",
                    options=sorted(df_full["Market"].dropna().unique())
                            if "Market" in df_full.columns else ["Unknown"],
                )
            with f3:
                item_total     = st.number_input("Order Item Total ($)", min_value=0.0, value=250.0, step=10.0)
                discount_rate  = st.number_input("Discount Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
                benefit        = st.number_input("Benefit per Order ($)", value=30.0, step=5.0)
                sales_cust     = st.number_input("Sales per Customer ($)", value=500.0, step=10.0)
                product_price  = st.number_input("Product Price ($)", value=200.0, step=10.0)

        predict_btn = st.button("Calculate Risk Score", type="primary")

        if predict_btn:
            # Build input frame matching raw feature structure.
            # Every object column that exists in the training data must be
            # present here — even ones not collected in the form — so the
            # feature engineering steps (target encoder, select_dtypes drop)
            # can process them identically to training.
            raw_input = {
                # ── user-supplied inputs ──────────────────────────
                "Shipping Mode":                  shipping_mode,
                "Order Item Quantity":            float(order_qty),
                "Days for shipment (scheduled)":  float(scheduled_days),
                "Order Region":                   order_region,
                "Customer Segment":               customer_segment,
                "Market":                         market,
                "Order Item Total":               item_total,
                "Order Item Discount Rate":       discount_rate,
                "Order Item Discount":            item_total * discount_rate,
                "Benefit per order":              benefit,
                "Sales per customer":             sales_cust,
                "Order Item Product Price":       product_price,
                "Product Price":                  product_price,
                "Sales":                          item_total,
                # ── columns present in training data, not in form ─
                # These are object columns dropped by select_dtypes
                # during training; supply "Unknown" so the encoder
                # maps them to the global mean and they vanish cleanly.
                "Order Country":                  "Unknown",
                "Order State":                    "Unknown",
                "Order City":                     "Unknown",
                "Order Status":                   "Unknown",
                "Customer City":                  "Unknown",
                "Customer Country":               "Unknown",
                "Customer State":                 "Unknown",
                "Department Name":                "Unknown",
                "Category Name":                  "Unknown",
                "Category Id":                    0,
                "Type":                           "DEBIT",
                # ── numeric defaults for any remaining train columns ─
                "Order Item Profit Ratio":        0.0,
            }

            X_new = pd.DataFrame([raw_input])

            # Impute missing numerics using training medians
            for col in X_new.select_dtypes(include=np.number).columns:
                X_new[col] = X_new[col].fillna(train_stats.get(col, 0.0))
            for col in X_new.select_dtypes(include="object").columns:
                X_new[col] = X_new[col].fillna("Unknown")

            # Engineered features (mirror pipeline exactly)
            X_new["shipping_pressure"] = (
                X_new["Order Item Quantity"] / (X_new["Days for shipment (scheduled)"] + 1)
            )
            qty_med = train_stats.get("Order Item Quantity", 5.0)
            X_new["high_quantity_flag"] = (X_new["Order Item Quantity"] > qty_med).astype(int)
            X_new["is_express"]         = 1 if "Express" in shipping_mode else 0
            X_new["order_value_per_qty"] = item_total / (order_qty + 1)
            X_new["discount_pressure"]   = discount_rate
            X_new["benefit_ratio"]       = benefit / (sales_cust + 1)
            sched_med = train_stats.get("Days for shipment (scheduled)", 4.0)
            X_new["schedule_tight"]      = 1 if scheduled_days < sched_med else 0

            bins = [-np.inf] + value_quantiles + [np.inf]
            X_new["order_value_tier"] = pd.cut(
                X_new["Order Item Total"], bins=bins,
                labels=[1, 2, 3, 4, 5]
            ).astype(float).fillna(3.0)

            X_new["route_key"] = market + "_" + shipping_mode

            # Target encode
            X_new = encoder.transform(X_new)
            X_new.drop(columns=["route_key"], inplace=True, errors="ignore")

            # One-hot
            ohe_cat_cols = [
                c for c in ["Shipping Mode", "Customer Segment", "Type", "Category Name"]
                if c in X_new.columns
            ]
            X_new = pd.get_dummies(X_new, columns=ohe_cat_cols, drop_first=True)
            X_new = X_new.select_dtypes(exclude="object")

            # ── Align to EXACTLY the columns the fitted model expects ──
            # Strategy: ask the classifier directly via feature_names_in_.
            # This is the ground truth — not ohe_columns.csv, which may
            # contain object columns that were later dropped by select_dtypes.
            # reindex() simultaneously:
            #   • drops columns unseen at fit time (e.g. Customer Country)
            #   • adds missing dummy columns as 0
            #   • enforces the exact column order the model was trained on
            try:
                clf_step = model.named_steps["clf"]
                expected_cols = clf_step.feature_names_in_.tolist()
            except AttributeError:
                # Fallback: ohe_cols is still a reasonable approximation
                expected_cols = ohe_cols

            X_new = X_new.reindex(columns=expected_cols, fill_value=0)

            try:
                prob = float(model.predict_proba(X_new)[0, 1])

                if prob >= 0.60:
                    risk_cat   = "High"
                    risk_color = COLOR_HIGH
                    badge_cls  = "badge-high"
                elif prob >= 0.30:
                    risk_cat   = "Medium"
                    risk_color = COLOR_MED
                    badge_cls  = "badge-medium"
                else:
                    risk_cat   = "Low"
                    risk_color = COLOR_LOW
                    badge_cls  = "badge-low"

                # Display result
                r1, r2, r3 = st.columns([1, 1, 2])

                with r1:
                    st.markdown(f"""
                    <div class='score-box'>
                      <div class='score-label'>Late Delivery Probability</div>
                      <div class='score-num' style='color:{risk_color}'>
                        {prob*100:.1f}<span style='font-size:24px'>%</span>
                      </div>
                      <div style='margin-top:12px'>
                        <span class='badge {badge_cls}'>{risk_cat} Risk</span>
                      </div>
                    </div>""", unsafe_allow_html=True)

                with r2:
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        number=dict(suffix="%", font=dict(size=28, color=risk_color)),
                        gauge=dict(
                            axis=dict(
                                range=[0, 100],
                                tickwidth=1,
                                tickcolor="#8B949E",
                                tickfont=dict(size=10),
                            ),
                            bar=dict(color=risk_color, thickness=0.3),
                            bgcolor=PLOTLY_BG,
                            borderwidth=0,
                            steps=[
                                dict(range=[0, 30],  color="rgba(63,185,80,0.15)"),
                                dict(range=[30, 60], color="rgba(227,169,69,0.15)"),
                                dict(range=[60, 100],color="rgba(255,107,107,0.15)"),
                            ],
                            threshold=dict(
                                line=dict(color="white", width=2),
                                thickness=0.85,
                                value=risk_threshold * 100,
                            ),
                        ),
                    ))
                    fig.update_layout(
                        paper_bgcolor=PLOTLY_PAPER_BG,
                        font=dict(color="#C9D1D9"),
                        height=260,
                        margin=dict(l=16, r=16, t=16, b=16),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with r3:
                    # Key drivers — shipping pressure + top factors
                    factors = {
                        "Shipping pressure index":   float(order_qty) / (scheduled_days + 1),
                        "Standard class risk flag":  1 - (1 if "Express" in shipping_mode or "Same Day" in shipping_mode else 0),
                        "Discount pressure":         discount_rate,
                        "Schedule tightness":        1 if scheduled_days < sched_med else 0,
                        "High quantity flag":        1 if order_qty > qty_med else 0,
                        "Order value per unit":      round(item_total / (order_qty + 1), 2),
                    }
                    factor_df = pd.DataFrame(
                        list(factors.items()), columns=["Factor", "Value"]
                    ).sort_values("Value", ascending=True)

                    fig = go.Figure(go.Bar(
                        x=factor_df["Value"],
                        y=factor_df["Factor"],
                        orientation="h",
                        marker=dict(
                            color=factor_df["Value"],
                            colorscale=[[0, COLOR_LOW], [0.5, COLOR_MED], [1, COLOR_HIGH]],
                        ),
                        hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
                    ))
                    fig = apply_dark_layout(fig, height=260, legend=False)
                    fig.update_layout(
                        title="Key Risk Factor Scores",
                        xaxis_title="Factor score",
                        yaxis_title="",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Recommendation
                if risk_cat == "High":
                    st.error(
                        "🚨 **High-risk order.** Recommend: upgrade shipping mode, "
                        "pre-alert customer, or flag for manual review."
                    )
                elif risk_cat == "Medium":
                    st.warning(
                        "⚠️ **Medium-risk order.** Monitor closely. "
                        "Consider buffer time or proactive customer update."
                    )
                else:
                    st.success("✅ **Low-risk order.** No immediate action required.")

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.info("Check that your feature columns match the training pipeline.")

    # ── Historical examples ─────────────────────────────────
    st.markdown(
        "<div class='section-header'>Sample High-Risk Orders (Test Set)</div>",
        unsafe_allow_html=True,
    )
    sample_cols = [c for c in [
        "Order Region", "Shipping Mode", "Customer Segment",
        "Order Item Quantity", "Days for shipment (scheduled)",
        "Late_Delivery_Probability", "Risk_Category", "Actual_Late",
    ] if c in df.columns]

    sample = (
        df[df["Risk_Category"] == "High"]
          .sort_values("Late_Delivery_Probability", ascending=False)
          [sample_cols]
          .head(10)
          .reset_index(drop=True)
    )
    st.dataframe(
        sample.style.background_gradient(
            subset=["Late_Delivery_Probability"],
            cmap="RdYlGn_r",
        ).format({"Late_Delivery_Probability": "{:.3f}"}),
        use_container_width=True,
        height=320,
    )


# ================================================================
# TAB 3 — REGION & MODE RISK ANALYSIS
# ================================================================

with tab3:
    st.markdown(
        "<div class='section-header'>Regional Risk Heatmap</div>",
        unsafe_allow_html=True,
    )

    if region_col:
        # Risk rate by region
        region_stats = (
            df.groupby(region_col)
              .agg(
                  avg_prob=("Late_Delivery_Probability", "mean"),
                  high_risk_count=("Risk_Category", lambda x: (x == "High").sum()),
                  total=("Late_Delivery_Probability", "count"),
              )
              .reset_index()
        )
        region_stats["high_risk_pct"] = (
            region_stats["high_risk_count"] / region_stats["total"] * 100
        )

        c1, c2 = st.columns([1, 1])

        with c1:
            fig = px.bar(
                region_stats.sort_values("avg_prob", ascending=True),
                x="avg_prob", y=region_col,
                orientation="h",
                color="avg_prob",
                color_continuous_scale=["#3FB950", "#E3A945", "#FF6B6B"],
                labels={"avg_prob": "Avg Risk Probability", region_col: ""},
                text="avg_prob",
                template=PLOTLY_TEMPLATE,
            )
            fig.update_traces(
                texttemplate="%{text:.2f}",
                textposition="outside",
                textfont=dict(size=10),
            )
            fig = apply_dark_layout(fig, height=480, legend=False)
            fig.update_layout(
                title=f"Avg Late Delivery Risk by {region_col}",
                coloraxis_showscale=False,
                xaxis=dict(range=[0, 1]),
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(
                region_stats.sort_values("high_risk_pct", ascending=True),
                x="high_risk_pct", y=region_col,
                orientation="h",
                color="high_risk_pct",
                color_continuous_scale=["#3FB950", "#E3A945", "#FF6B6B"],
                labels={"high_risk_pct": "High-Risk %", region_col: ""},
                text="high_risk_pct",
                template=PLOTLY_TEMPLATE,
            )
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
                textfont=dict(size=10),
            )
            fig = apply_dark_layout(fig, height=480, legend=False)
            fig.update_layout(
                title=f"High-Risk Order % by {region_col}",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='section-header'>Shipping Mode Risk Comparison</div>",
        unsafe_allow_html=True,
    )

    if "Shipping Mode" in df.columns:
        mode_stats = (
            df.groupby("Shipping Mode")
              .agg(
                  avg_prob=("Late_Delivery_Probability", "mean"),
                  high_pct=("Risk_Category", lambda x: (x == "High").mean() * 100),
                  med_pct=("Risk_Category", lambda x: (x == "Medium").mean() * 100),
                  low_pct=("Risk_Category", lambda x: (x == "Low").mean() * 100),
                  count=("Late_Delivery_Probability", "count"),
              )
              .reset_index()
        )

        c3, c4 = st.columns([1, 1])

        with c3:
            fig = go.Figure()
            for col, name, color in [
                ("high_pct", "High", COLOR_HIGH),
                ("med_pct",  "Medium", COLOR_MED),
                ("low_pct",  "Low",  COLOR_LOW),
            ]:
                fig.add_trace(go.Bar(
                    name=name,
                    x=mode_stats["Shipping Mode"],
                    y=mode_stats[col],
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate=f"<b>{name}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
                ))
            fig = apply_dark_layout(fig, height=360)
            fig.update_layout(
                title="Risk Category Distribution by Shipping Mode",
                barmode="stack",
                xaxis_title="",
                yaxis_title="% of Orders",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            # Scatter: avg probability vs order count by mode
            fig = go.Figure()
            colors_scatter = [COLOR_HIGH, COLOR_MED, COLOR_LOW, COLOR_BLUE]
            for i, row in mode_stats.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["count"]],
                    y=[row["avg_prob"] * 100],
                    mode="markers+text",
                    name=row["Shipping Mode"],
                    text=[row["Shipping Mode"]],
                    textposition="top center",
                    textfont=dict(size=10),
                    marker=dict(
                        size=max(12, row["count"] / mode_stats["count"].max() * 60),
                        color=colors_scatter[i % len(colors_scatter)],
                        opacity=0.85,
                        line=dict(width=1, color="#21262D"),
                    ),
                    hovertemplate=(
                        f"<b>{row['Shipping Mode']}</b><br>"
                        f"Orders: {row['count']:,}<br>"
                        f"Avg risk: {row['avg_prob']*100:.1f}%<extra></extra>"
                    ),
                ))
            fig = apply_dark_layout(fig, height=360, legend=False)
            fig.update_layout(
                title="Volume vs Avg Risk by Mode",
                xaxis_title="Order count",
                yaxis_title="Avg risk probability (%)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Region × Mode risk matrix
    st.markdown(
        "<div class='section-header'>Region × Shipping Mode Risk Matrix</div>",
        unsafe_allow_html=True,
    )

    if region_col and "Shipping Mode" in df.columns:
        pivot = (
            df.groupby([region_col, "Shipping Mode"])["Late_Delivery_Probability"]
              .mean()
              .unstack(fill_value=0)
        )
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#1A2F1A"], [0.5, "#5C3D10"], [1, "#5C1A1A"]],
            text=[[f"{v:.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(
                title="Avg prob",
                tickfont=dict(color="#C9D1D9"),
                title_font=dict(color="#C9D1D9"),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>Mode: %{x}<br>Avg risk: %{z:.3f}<extra></extra>"
            ),
        ))
        fig = apply_dark_layout(fig, height=420, legend=False)
        fig.update_layout(
            title="Avg Late Delivery Probability — Region × Shipping Mode",
            xaxis_title="Shipping Mode",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
# TAB 4 — OPERATIONS ACTION PANEL
# ================================================================

with tab4:
    st.markdown(
        "<div class='section-header'>Operations Attention Queue</div>",
        unsafe_allow_html=True,
    )

    # Filter controls
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        panel_threshold = st.slider(
            "Risk threshold", 0.10, 0.90,
            value=float(risk_threshold), step=0.05, key="panel_thr",
        )
    with a2:
        top_n_orders = st.select_slider(
            "Show top N orders",
            options=[25, 50, 100, 200, 500, "All"],
            value=100,
        )
    with a3:
        sort_metric = st.selectbox(
            "Sort by",
            options=["Late_Delivery_Probability", "Order Item Quantity",
                     "Days for shipment (scheduled)"],
        )
    with a4:
        show_only_high = st.checkbox("High risk only", value=True)

    # Build action queue
    action_df = df[
        df["Late_Delivery_Probability"] >= panel_threshold
    ].copy() if show_only_high else df[
        df["Late_Delivery_Probability"] >= panel_threshold
    ].copy()

    if sort_metric in action_df.columns:
        action_df = action_df.sort_values(sort_metric, ascending=False)

    if top_n_orders != "All":
        action_df = action_df.head(int(top_n_orders))

    # Summary row
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Orders in queue", f"{len(action_df):,}")
    s2.metric(
        "Avg probability",
        f"{action_df['Late_Delivery_Probability'].mean()*100:.1f}%",
    )
    if "Actual_Late" in action_df.columns:
        s3.metric("True delays captured",
                  f"{action_df['Actual_Late'].mean()*100:.1f}%")
    if "Days for shipment (scheduled)" in action_df.columns:
        s4.metric("Avg scheduled days",
                  f"{action_df['Days for shipment (scheduled)'].mean():.1f}")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Probability distribution for queue
    c5, c6 = st.columns([1, 1])

    with c5:
        fig = go.Figure(go.Histogram(
            x=action_df["Late_Delivery_Probability"],
            nbinsx=30,
            marker=dict(
                color=action_df["Late_Delivery_Probability"],
                colorscale=[[0, COLOR_MED], [1, COLOR_HIGH]],
                line=dict(color=PLOTLY_PAPER_BG, width=0.5),
            ),
            hovertemplate="Probability: %{x:.2f}<br>Count: %{y}<extra></extra>",
        ))
        fig = apply_dark_layout(fig, height=280, legend=False)
        fig.update_layout(
            title="Action Queue — Risk Distribution",
            xaxis_title="Late Delivery Probability",
            yaxis_title="Order count",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        if "Shipping Mode" in action_df.columns:
            mode_counts = (
                action_df.groupby(["Shipping Mode", "Risk_Category"])
                  .size().reset_index(name="count")
            )
            fig = px.bar(
                mode_counts,
                x="Shipping Mode", y="count", color="Risk_Category",
                color_discrete_map=RISK_COLORS,
                barmode="group",
                template=PLOTLY_TEMPLATE,
                labels={"count": "Orders", "Shipping Mode": ""},
            )
            fig = apply_dark_layout(fig, height=280)
            fig.update_layout(title="Queue — Mode × Risk Category")
            st.plotly_chart(fig, use_container_width=True)

    # ── Action table ────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Prioritised Order List</div>",
        unsafe_allow_html=True,
    )

    display_cols = [c for c in [
        "Order Region", "Market", "Shipping Mode",
        "Customer Segment", "Order Item Quantity",
        "Days for shipment (scheduled)", "Order Item Total",
        "Late_Delivery_Probability", "Risk_Category", "Actual_Late",
    ] if c in action_df.columns]

    display_df = action_df[display_cols].reset_index(drop=True)
    display_df.index = display_df.index + 1   # 1-based ranking

    # Export button
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Export queue as CSV",
        data=csv_bytes,
        file_name="high_risk_orders.csv",
        mime="text/csv",
    )

    # Styled dataframe
    def style_risk(val):
        if val == "High":
            return "background-color:rgba(255,107,107,0.15);color:#FF6B6B;font-weight:600"
        elif val == "Medium":
            return "background-color:rgba(227,169,69,0.15);color:#E3A945;font-weight:600"
        elif val == "Low":
            return "background-color:rgba(63,185,80,0.15);color:#3FB950;font-weight:600"
        return ""

    styled = (
        display_df.style
          .applymap(style_risk, subset=["Risk_Category"])
          .background_gradient(
              subset=["Late_Delivery_Probability"],
              cmap="RdYlGn_r", vmin=0, vmax=1,
          )
          .format({
              "Late_Delivery_Probability": "{:.3f}",
              "Order Item Total": "${:,.2f}" if "Order Item Total" in display_cols else "{}",
          })
    )
    st.dataframe(styled, use_container_width=True, height=460)

    # ── Priority actions chart ───────────────────────────────────
    st.markdown(
        "<div class='section-header'>Intervention Priority</div>",
        unsafe_allow_html=True,
    )

    if region_col in action_df.columns and "Shipping Mode" in action_df.columns:
        priority = (
            action_df.groupby([region_col, "Shipping Mode"])
              .agg(
                  orders=("Late_Delivery_Probability", "count"),
                  avg_risk=("Late_Delivery_Probability", "mean"),
              )
              .reset_index()
              .sort_values("avg_risk", ascending=False)
              .head(15)
        )

        fig = px.scatter(
            priority,
            x="orders", y="avg_risk",
            color="avg_risk",
            size="orders",
            hover_name=region_col,
            hover_data=["Shipping Mode", "orders"],
            color_continuous_scale=[[0, COLOR_MED], [1, COLOR_HIGH]],
            labels={
                "orders": "High-risk order count",
                "avg_risk": "Avg risk probability",
            },
            text=region_col,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_traces(
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(opacity=0.85),
        )
        fig = apply_dark_layout(fig, height=380, legend=False)
        fig.update_layout(
            title="Region × Mode: Volume vs Avg Risk (bubble = order count)",
            coloraxis_showscale=False,
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# 5 — FOOTER
# ================================================================

st.markdown("""
<hr>
<div style='text-align:center;padding:16px 0 8px;font-size:11px;
            color:#8B949E;letter-spacing:0.5px'>
  APL Logistics  ·  Delivery Risk Intelligence Platform
  ·  Powered by Random Forest + XGBoost + SHAP
  ·  Models retrain on latest data automatically
</div>
""", unsafe_allow_html=True)

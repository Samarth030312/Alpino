"""
Alpino AI Operating System — Executive Dashboard
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Alpino AI Operating System",
    page_icon="🥜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Alpino AI Operating System")
st.markdown("---")

# ---------------------------------------------------------------------------
# Top-Line Impact Metrics
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Financial Disparities Caught", "25%", delta="12 of 50 invoices",
          help="Validation engine detects 100% of injected errors")
m2.metric("Employees at High Flight Risk", "35", delta="of 100 scored",
          delta_color="inverse",
          help="XGBoost attrition model flags immediate intervention")
m3.metric("Predicted Campaign Stockouts", "1", delta="Day 15 spike",
          delta_color="inverse",
          help="Meta campaign demand (200 units) exceeds inventory (150)")
m4.metric("Projected ROAS Lift", "3.16 → 4.1x",
          delta="+29%",
          help="Via RFM-based budget reallocation to high-LTV segments")

st.markdown("---")

# ---------------------------------------------------------------------------
# Three Pillars — Consultant Insight Boxes
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("### Finance")
    st.info(
        "**Objective:** Reduce 1-2 month reconciliation lag to days\n\n"
        "**Mechanism:** 3-stage OCR pipeline — extract, normalize, validate\n\n"
        "**ROI:** Rs. 5-6L/yr in labor savings + revenue leakage recovery"
    )
    st.page_link("pages/1_Finance_OCR.py", label="Open Finance OCR", icon="📄")

with col2:
    st.markdown("### HR")
    st.info(
        "**Objective:** Cut 19.44% early attrition to <10%\n\n"
        "**Mechanism:** XGBoost risk scoring + SBERT resume screening\n\n"
        "**ROI:** Rs. 3-4L/yr saved on wasted onboarding across 36 hires"
    )
    st.page_link("pages/2_HR_Analytics.py", label="Open HR Analytics", icon="👥")

with col3:
    st.markdown("### Marketing")
    st.info(
        "**Objective:** Eliminate promotional stockouts, lift ROAS 29%\n\n"
        "**Mechanism:** RFM segmentation + demand-inventory alignment\n\n"
        "**ROI:** Protect Rs. 13.14L paid social revenue from CAC waste"
    )
    st.page_link("pages/3_Marketing_Insights.py", label="Open Marketing", icon="📊")

st.markdown("---")

# ---------------------------------------------------------------------------
# Implementation Roadmap
# ---------------------------------------------------------------------------
st.subheader("Implementation Roadmap")

r1, r2, r3 = st.columns(3)
with r1:
    st.success(
        "**Phase 1 — Foundation (M1-3)**\n\n"
        "- Invoice digitization pilot (top 5 vendors)\n"
        "- SBERT resume screening for Marketing\n"
        "- CDP onboarding\n\n"
        "Budget: Rs. 7-12L"
    )
with r2:
    st.warning(
        "**Phase 2 — Integration (M4-6)**\n\n"
        "- Full vendor OCR + Tally-UniCommerce middleware\n"
        "- XGBoost attrition dashboards\n"
        "- Demand-inventory linking\n\n"
        "Budget: Rs. 9-14L"
    )
with r3:
    st.info(
        "**Phase 3 — Intelligence (M7-12)**\n\n"
        "- Predictive cash flow\n"
        "- Dynamic pricing + campaign automation\n"
        "- Proactive talent strategy\n\n"
        "**Total Year-1: Rs. 25-40L | Payback: 12-18 months**"
    )
